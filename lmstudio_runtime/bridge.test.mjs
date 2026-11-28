import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { once } from "node:events";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { nativeSession } from "./bridge.mjs";

const require = createRequire(import.meta.url);
const { WebSocketServer } = require(join(dirname(require.resolve("ws/package.json")), "index.js"));

const mode = process.argv[2] ?? "normal";
const cases = new Set(["normal", "tool", "truncated", "unclosed", "cancel", "ambiguous", "changed"]);
assert.ok(cases.has(mode));
const watchdog = setTimeout(() => process.exit(1), 10000);
watchdog.unref();

const identity = {
    modelKey: "test/model", format: "gguf", path: "/test/model.gguf", sizeBytes: 456,
    deviceIdentifier: null, identifier: "test-instance", instanceReference: "test-reference",
};
const info = {
    ...identity, type: "llm", displayName: "Test Model", publisher: "test", indexedModelIdentifier: "test/model",
    ttlMs: null, lastUsedTime: null, vision: false, trainedForToolUse: false,
    maxContextLength: 4096, contextLength: 64,
};
const messages = [{ role: "user", content: " π \r\n  Snowman ☃ " }];
const marker = "\n<|assistant|>\n";
const rendered = "<s>\nUser: " + messages[0].content + marker;
const specifier = { type: "instanceReference", instanceReference: info.instanceReference };
const loadConfig = { fields: [{ key: "llm.load.contextLength", value: 64 }] };
const predictionConfig = { fields: [
    { key: "llm.prediction.maxPredictedTokens", value: { checked: true, value: 4 } },
    { key: "llm.prediction.contextOverflowPolicy", value: "stopAtLimit" },
    { key: "llm.prediction.stopStrings", value: [] },
] };
const requests = [];
let armed = false;
let disarmed = false;
let dispatched;
let cancelled;
const began = new Promise(resolve => { dispatched = resolve; });
const cancelledOnWire = new Promise(resolve => { cancelled = resolve; });
const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
await once(wss, "listening");
wss.on("connection", (ws, request) => {
    assert.equal(request.url, "/llm");
    let authenticated = false;
    ws.on("message", raw => {
        const msg = JSON.parse(raw.toString());
        const send = packet => ws.send(JSON.stringify(packet));
        if (!authenticated) {
            authenticated = true;
            send({ success: true });
            return;
        }
        requests.push(msg);
        if (msg.type === "rpcCall") {
            const result = {
                listLoaded: mode === "ambiguous" ? [info, info] : [info],
                applyPromptTemplate: { formatted: rendered },
                tokenize: { tokens: [1, 2, 3] },
                getModelInfo: info,
                getLoadConfig: loadConfig,
            }[msg.endpoint];
            assert.ok(result, msg.endpoint);
            if (msg.endpoint !== "listLoaded") assert.deepEqual(msg.parameter.specifier, specifier);
            send({ type: "rpcResult", callId: msg.callId, result });
        }
        if (msg.type === "channelSend" && msg.message.type === "cancel") {
            if (mode === "cancel") {
                send({ type: "channelSend", channelId: msg.channelId, message: {
                    type: "success", stats: { stopReason: "userStopped" },
                    modelInfo: info, loadModelConfig: loadConfig, predictionConfig,
                } });
                send({ type: "channelClose", channelId: msg.channelId });
            }
            cancelled();
        }
        if (msg.type !== "channelCreate") return;
        dispatched();
        if (mode === "cancel") return;
        const event = message => send({ type: "channelSend", channelId: msg.channelId, message });
        const fragment = (content, reasoningType) => event({ type: "fragment", fragment: {
            content, tokensCount: 1, containsDrafted: false, reasoningType, isStructural: false,
        } });
        if (mode === "tool") event({ type: "toolCallGenerationStart", toolCallId: "t" });
        fragment("<think>", "reasoningStartTag");
        fragment("private", "reasoning");
        fragment("</think>", "reasoningEndTag");
        fragment("exact answer", "none");
        event({
            type: "success",
            stats: { stopReason: "eosFound", promptTokensCount: 4, predictedTokensCount: 4,
                ...(mode === "truncated" ? { truncated: true } : {}) },
            modelInfo: mode === "changed" ? { ...info, instanceReference: "other" } : info,
            loadModelConfig: loadConfig, predictionConfig,
        });
        if (mode !== "unclosed") send({ type: "channelClose", channelId: msg.channelId });
    });
});

let session;
try {
    session = await nativeSession({ baseUrl: "ws://127.0.0.1:" + wss.address().port,
        idle: async () => true, arm: async () => { armed = true; },
        disarm: async () => { disarmed = true; } });
    const inspect = session.inspect({ identity, probe: messages, generationMarker: marker,
        probeSha256: createHash("sha256").update(rendered).digest("hex"),
        context: 64, admitContext: 64, framingMax: 1, outputLimit: 8 });
    if (mode === "ambiguous") {
        await assert.rejects(inspect, { code: "model_unavailable" });
        assert.equal(requests.filter(row => row.endpoint === "completeRawText").length, 0);
    } else {
        assert.equal((await inspect).effectiveContext, 64);
        const prepared = await session.prepare({ messages });
        assert.equal(prepared.rendered, rendered);
        assert.equal(prepared.nativeTokens, 3);
        const prediction = session.predict({ promptSha256: prepared.promptSha256, maxTokens: 4, answerBytes: 128 });
        if (mode === "normal") {
            const result = await prediction;
            assert.equal(result.text, "exact answer");
            assert.equal(result.stopReason, "eosFound");
            assert.equal(result.effectiveContext, 64);
            assert.equal(result.overflow, "stopAtLimit");
            assert.equal(result.maxTokens, 4);
            assert.equal(result.promptTokens, 4);
            assert.ok(disarmed);
        } else {
            if (mode === "cancel") {
                await began;
                session.abort();
            }
            const code = ({ tool: "unsupported_tool_output", truncated: "incomplete",
                unclosed: "incomplete", cancel: "cancelled", changed: "model_unavailable" })[mode];
            await assert.rejects(prediction, { code });
            if (mode === "cancel") await cancelledOnWire;
            assert.equal(disarmed, false);
        }
        assert.ok(armed);
        const renders = requests.filter(row => row.endpoint === "applyPromptTemplate");
        assert.equal(renders.length, 2);
        for (const row of renders) {
            assert.equal(row.parameter.opts.omitBosToken, false);
            assert.equal(row.parameter.opts.omitEosToken, false);
        }
        const tokenize = requests.filter(row => row.endpoint === "tokenize");
        const dispatch = requests.filter(row => row.endpoint === "completeRawText");
        assert.equal(tokenize.length, 1);
        assert.equal(dispatch.length, 1);
        const raw = dispatch[0].creationParameter;
        assert.deepEqual(Buffer.from(tokenize[0].parameter.inputString), Buffer.from(raw.rawPrompt));
        assert.deepEqual(Buffer.from(rendered), Buffer.from(raw.rawPrompt));
        assert.deepEqual(raw.modelSpecifier, specifier);
        assert.equal(raw.fuzzyPresetIdentifier, undefined);
        assert.ok(raw.predictionConfigStack.layers.every(layer => layer.config.fields.every(
            field => !/tool|promptTemplate|systemPrompt/i.test(field.key))));
    }
    process.stdout.write("PASS " + mode + "\n");
} finally {
    await session?.close();
    for (const ws of wss.clients) ws.terminate();
    wss.close();
    clearTimeout(watchdog);
}
