import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { createInterface } from "node:readline";
import { setTimeout as delay } from "node:timers/promises";
import { promisify } from "node:util";

const execute = promisify(execFile);
const LOOPBACK = "ws://127.0.0.1:1234";
const FRAME_LIMIT = 8 * 1024 * 1024;
const CONTEXT = "llm.load.contextLength";
const OVERFLOW = "llm.prediction.contextOverflowPolicy";
const OUTPUT = "llm.prediction.maxPredictedTokens";
const PROBE_DEFAULTS = { omitBosToken: false, omitEosToken: false };
const digest = text => createHash("sha256").update(text, "utf8").digest("hex");
const require = createRequire(import.meta.url);

function check(condition, code = "invalid_response") {
    if (!condition) throw new Failure(code);
}

class Failure extends Error {
    constructor(code) {
        super(code);
        this.code = code;
    }
}

function sameIdentity(info, identity) {
    return info && Object.keys(identity).every(key => Object.hasOwn(info, key) && info[key] === identity[key]);
}

function configValue(config, key) {
    const matches = config?.fields?.filter(field => field.key === key);
    return matches?.length === 1 ? matches[0].value : undefined;
}

async function isIdle(lms, identity) {
    try {
        const { stdout } = await execute(lms, ["ps", "--json"], {
            timeout: 5000, maxBuffer: 1024 * 1024, windowsHide: true,
        });
        const rows = JSON.parse(stdout).filter(row => row.identifier === identity.identifier);
        return rows.length === 1 && rows[0].status === "idle" && rows[0].queued === 0;
    } catch {
        return false;
    }
}

const statsKeys = new Set([
    "stopReason", "stopString", "tokensPerSecond", "numGpuLayers", "timeToFirstTokenSec",
    "totalTimeSec", "promptTokensCount", "predictedTokensCount", "totalTokensCount",
    "usedDraftModelKey", "totalDraftTokensCount", "acceptedDraftTokensCount",
    "rejectedDraftTokensCount", "ignoredDraftTokensCount",
]);
const successKeys = new Set(["type", "stats", "modelInfo", "loadModelConfig", "predictionConfig"]);
const rpcEndpoints = new Set(["listLoaded", "getModelInfo", "getLoadConfig", "applyPromptTemplate", "tokenize"]);

// Only the optional isolated Node process uses this observer. The browser
// export of the pinned SDK uses Node's WebSocket; the normal Node export
// would strip unknown fields before they reach the public PredictionResult.
function observeTransport(baseUrl, fail) {
    // Use the same Node WebSocket implementation as the SDK's regular
    // export, while the browser export lets this isolated process observe it.
    const Original = require(join(dirname(require.resolve("ws/package.json")), "index.js"));
    const state = { socket: undefined, reference: undefined, lastTokenized: undefined,
        prediction: undefined, loadCall: undefined, context: undefined };
    globalThis.WebSocket = class extends Original {
        constructor(address) {
            check(address === baseUrl + "/llm");
            super(address, { followRedirects: false, maxPayload: FRAME_LIMIT });
            state.socket = this;
            this.addEventListener("message", event => {
                try {
                    const packet = JSON.parse(event.data);
                    if (packet.type === "rpcResult" && packet.callId === state.loadCall) {
                        state.context = configValue(packet.result, CONTEXT);
                        state.loadCall = undefined;
                    }
                    if (state.prediction) receive(state.prediction, packet);
                } catch (error) {
                    fail(error.code ?? "invalid_response");
                }
            });
        }

        send(data) {
            const packet = JSON.parse(data);
            if (packet.type === "rpcCall") {
                check(rpcEndpoints.has(packet.endpoint));
                if (packet.endpoint !== "listLoaded") {
                    check(packet.parameter.specifier.type === "instanceReference"
                        && packet.parameter.specifier.instanceReference === state.reference);
                }
                if (packet.endpoint === "tokenize") state.lastTokenized = packet.parameter.inputString;
                if (packet.endpoint === "getLoadConfig") {
                    state.loadCall = packet.callId;
                    state.context = undefined;
                }
            } else if (packet.type === "channelCreate") {
                const call = state.prediction;
                const parameter = packet.creationParameter;
                check(call && call.id === undefined && packet.endpoint === "completeRawText");
                check(parameter.rawPrompt === call.prompt && call.prompt === state.lastTokenized
                    && parameter.modelSpecifier.type === "instanceReference"
                    && parameter.modelSpecifier.instanceReference === call.reference
                    && parameter.fuzzyPresetIdentifier === undefined);
                call.id = packet.channelId;
            } else if (packet.type === "communicationWarning") {
                // Invalid SDK frames already reach the local logger. Do not
                // send the SDK's diagnostic payload back over the wire.
                fail("invalid_response");
                return;
            } else if (packet.type === "channelSend") {
                check(packet.channelId === state.prediction?.id && packet.message?.type === "cancel");
            } else {
                check(packet.type === undefined && typeof packet.clientIdentifier === "string");
            }
            super.send(data);
        }
    };
    return state;
}

function receive(call, packet) {
    if (call.id === undefined || packet.channelId !== call.id) return;
    if (packet.type === "channelClose") {
        check(!call.closed, "incomplete");
        call.closed = true;
        // Let the SDK consume the same closure before disposing its channel.
        setImmediate(() => call.close(call.success));
        return;
    }
    check(packet.type === "channelSend");
    const event = packet.message;
    if (typeof event?.type === "string" && event.type.toLowerCase().includes("tool")) {
        throw new Failure("unsupported_tool_output");
    }
    check(!call.success, "incomplete");
    if (event?.type === "success") {
        check(event.stats && Object.keys(event).every(key => successKeys.has(key))
            && Object.keys(event.stats).every(key => statsKeys.has(key)), "incomplete");
        call.success = true;
    } else {
        check(event?.type === "fragment" || event?.type === "promptProcessingProgress");
        check(!Object.keys(event).some(key => /truncat|tool/i.test(key))
            && !Object.keys(event.fragment ?? {}).some(key => /truncat|tool/i.test(key)));
    }
}

export async function nativeSession({ baseUrl = LOOPBACK, idle, arm, disarm }) {
    // An alternate port is available to the fake peer only. Production IPC
    // does not accept an endpoint or runtime options.
    check(/^ws:\/\/127\.0\.0\.1:[0-9]{1,5}$/.test(baseUrl));
    let failed;
    let controller;
    let abortNotice;
    const fail = code => {
        if (failed === undefined) failed = code;
        controller?.abort();
        abortNotice?.();
    };
    const wire = observeTransport(baseUrl, fail);
    const packageJson = new URL("../package.json", import.meta.resolve("@lmstudio/sdk"));
    check(JSON.parse(await readFile(packageJson, "utf8")).version === "2.1.0");
    const { LMStudioClient } = await import("@lmstudio/sdk");
    const client = new LMStudioClient({
        baseUrl, logger: { debug() {}, info() {}, warn() { fail("invalid_response"); },
            error() { fail("invalid_response"); } },
    });
    let deployment;
    let handle;
    let prepared;

    function healthy() {
        if (failed !== undefined) throw new Failure(failed);
    }

    async function assess() {
        healthy();
        const info = await handle.getModelInfo();
        const effective = await handle.getContextLength();
        healthy();
        check(sameIdentity(info, deployment.identity), "model_unavailable");
        check(wire.context === effective && info.contextLength === effective && effective === deployment.context
            && Number.isSafeInteger(info.maxContextLength) && effective <= info.maxContextLength);
        return info;
    }

    return {
        abort() { fail("cancelled"); },
        async close() {
            const socket = wire.socket;
            if (socket && (failed !== undefined || (wire.prediction && !wire.prediction.closed))) {
                // SDK 2.1.0 can report a second completion to the iterator if
                // transport teardown follows success without channel closure.
                // This isolated session is abandoned and the lease owner
                // independently checks runner idleness before releasing it.
                socket.removeAllListeners();
                socket.on("error", () => {});
                socket.terminate();
                return;
            }
            await Promise.race([client[Symbol.asyncDispose](), delay(1000, undefined, { ref: false })]);
        },
        async inspect(body) {
            check(deployment === undefined);
            check(body.identity?.deviceIdentifier === null && body.identity?.format === "gguf");
            check(Number.isSafeInteger(body.admitContext) && body.admitContext > 0
                && body.admitContext <= body.context);
            check(await idle(body.identity), "resource_busy");
            const matches = (await client.llm.listLoaded()).filter(model => model.identifier === body.identity.identifier);
            check(matches.length === 1 && matches[0].instanceReference === body.identity.instanceReference,
                "model_unavailable");
            deployment = body;
            handle = matches[0];
            wire.reference = body.identity.instanceReference;
            const info = await assess();
            const rendered = await handle.applyPromptTemplate(body.probe, PROBE_DEFAULTS);
            healthy();
            check(wire.socket && rendered.endsWith(body.generationMarker)
                && digest(rendered) === body.probeSha256);
            return { ...info, effectiveContext: info.contextLength };
        },
        async prepare(body) {
            const info = await assess();
            const rendered = await handle.applyPromptTemplate(body.messages, PROBE_DEFAULTS);
            check(Buffer.byteLength(rendered, "utf8") < FRAME_LIMIT / 8
                && rendered.endsWith(deployment.generationMarker));
            const ids = await handle.tokenize(rendered);
            healthy();
            check(ids.length > 0 && ids.every(id => Number.isSafeInteger(id) && id >= 0));
            prepared = { rendered, promptSha256: digest(rendered), nativeTokens: ids.length,
                effectiveContext: info.contextLength };
            return prepared;
        },
        async predict(body) {
            await assess();
            check(prepared && prepared.promptSha256 === body.promptSha256 && wire.prediction === undefined);
            check(Number.isSafeInteger(body.maxTokens) && body.maxTokens > 0 && body.maxTokens <= deployment.outputLimit
                && prepared.nativeTokens + deployment.framingMax + body.maxTokens <= deployment.admitContext
                && Number.isSafeInteger(body.answerBytes) && body.answerBytes > 0 && body.answerBytes <= 65536);
            check(await idle(deployment.identity), "resource_busy");
            await arm(deployment.identity);
            healthy();

            controller = new AbortController();
            let close;
            const closure = new Promise(resolve => { close = resolve; });
            const aborted = new Promise(resolve => { abortNotice = () => resolve(false); });
            const call = { id: undefined, prompt: prepared.rendered,
                reference: deployment.identity.instanceReference, close, success: false, closed: false };
            wire.prediction = call;
            const prediction = handle.complete(prepared.rendered, {
                maxTokens: body.maxTokens, contextOverflowPolicy: "stopAtLimit", signal: controller.signal,
            });
            // SDK predictions buffer their own fragments, so an abort also
            // closes the transport if a peer continues sending after cancel.
            const consumed = readAnswer(prediction, body.answerBytes);
            consumed.catch(error => fail(error.code ?? "invalid_response"));
            const result = await Promise.race([consumed, aborted]);
            if (failed !== undefined && call.id !== undefined) {
                await Promise.race([closure, delay(1500)]);
            }
            healthy();

            let timer;
            try {
                const timeout = new Promise(resolve => { timer = setTimeout(() => resolve(false), 3000); });
                check(await Promise.race([closure, aborted, timeout]), "incomplete");
            } finally {
                clearTimeout(timer);
            }
            healthy();
            const { receipt, text } = result;
            const info = await assess();
            check(sameIdentity(receipt.modelInfo, deployment.identity)
                && receipt.modelInfo.contextLength === info.contextLength, "model_unavailable");
            const outputLimit = configValue(receipt.predictionConfig, OUTPUT);
            const stopStrings = configValue(receipt.predictionConfig, "llm.prediction.stopStrings");
            check(Array.isArray(stopStrings) && stopStrings.length === 0);
            check(outputLimit?.checked === true);
            await disarm();
            healthy();
            return {
                text, identity: receipt.modelInfo, stopReason: receipt.stats.stopReason,
                promptTokens: receipt.stats.promptTokensCount, outputTokens: receipt.stats.predictedTokensCount,
                maxTokens: outputLimit.value, overflow: configValue(receipt.predictionConfig, OVERFLOW),
                effectiveContext: configValue(receipt.loadConfig, CONTEXT),
            };
        },
    };
}

async function readAnswer(prediction, limit) {
    const chunks = [];
    let bytes = 0;
    let received = 0;
    let reasoning = false;
    for await (const fragment of prediction) {
        received += Buffer.byteLength(fragment.content, "utf8") + 1;
        check(received <= FRAME_LIMIT / 8);
        switch (fragment.reasoningType) {
        case "reasoningStartTag":
            check(!reasoning);
            reasoning = true;
            break;
        case "reasoningEndTag":
            check(reasoning);
            reasoning = false;
            break;
        case "reasoning":
            check(reasoning || !fragment.isStructural);
            break;
        case "none":
            check(!reasoning && !fragment.isStructural);
            bytes += Buffer.byteLength(fragment.content, "utf8");
            check(bytes <= limit);
            chunks.push(fragment.content);
            break;
        default:
            throw new Failure("invalid_response");
        }
    }
    check(!reasoning, "incomplete");
    return { receipt: await prediction.result(), text: chunks.join("") };
}

async function run(marker, lms) {
    const session = await nativeSession({
        idle: identity => isIdle(lms, identity),
        arm: identity => writeFile(marker, JSON.stringify(identity), { flag: "wx" }),
        disarm: () => rm(marker),
    });
    const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
    let pending = false;
    let closed = false;
    const close = async () => {
        if (closed) return;
        closed = true;
        if (pending) session.abort();
        input.close();
        if (!pending) await session.close();
    };
    input.on("close", () => { void close(); });
    input.on("line", async line => {
        if (closed) return;
        if (pending || Buffer.byteLength(line) > FRAME_LIMIT) {
            await close();
            return;
        }
        pending = true;
        let command;
        let timer;
        try {
            const { command: name, ...body } = JSON.parse(line);
            check(["inspect", "prepare", "predict"].includes(name));
            command = name;
            // Client deadlines close stdin. This timeout also covers a
            // cancelled or unresponsive RPC without holding up lease recovery.
            const timeout = new Promise((_, reject) => {
                timer = setTimeout(() => reject(new Failure("timeout")), name === "predict" ? 300000 : 20000);
            });
            const value = await Promise.race([session[name](body), timeout]);
            if (!closed) {
                const reply = JSON.stringify({ command, value });
                check(Buffer.byteLength(reply) < FRAME_LIMIT);
                process.stdout.write(reply + "\n");
            }
        } catch (error) {
            if (!closed && command) {
                const code = error instanceof Failure ? error.code : "provider_error";
                process.stdout.write(JSON.stringify({ command, code }) + "\n");
            }
            await close();
        } finally {
            clearTimeout(timer);
            pending = false;
            if (closed) await session.close();
        }
    });
}

async function main() {
    const [mode, marker, lms] = process.argv.slice(2);
    if (mode === "--recover") {
        const identity = JSON.parse(await readFile(marker, "utf8"));
        process.exitCode = await isIdle(lms, identity) ? 0 : 75;
    } else if (mode === "--run") {
        await run(marker, lms);
    } else {
        process.exitCode = 1;
    }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    // Do not let SDK exceptions or prompt fields escape into diagnostics.
    main().catch(() => { process.exitCode = 1; });
}
