"""Opt-in LM Studio generation on the maintained Chroma Qt path."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

from generation_models import (
    GenerationError, GenerationProfile, _GenerationRouter,
    _raise_if_cancelled, _safe_model_id,
)
from token_budget import TokenCount, preserves_messages, text_messages, unknown


RENDER_PROBE = [
    {"role": "system", "content": "Native template verification."},
    {"role": "user", "content": "First turn: π.\n  Retain whitespace."},
    {"role": "assistant", "content": "Earlier answer."},
    {"role": "user", "content": "Last turn."},
]
_MAX_FRAME = 8 * 1024 * 1024
_RUNTIME_DIR = Path(__file__).resolve().with_name("lmstudio_runtime")


def _fail(code, message):
    return GenerationError(code, message, provider="lmstudio")


def _sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Deployment:
    """A binding derived from an independently verified deployment receipt.

    The normal app factory supplies no deployment. Only the separately
    approved smoke process may inject this type until enrolment is approved.
    """

    model_id: str
    model_key: str
    model_path: str
    size_bytes: int
    instance_reference: str
    context_tokens: int
    output_tokens: int
    generation_marker: str
    render_probe_sha256: str
    framing_min: int
    framing_max: int
    evidence_sha256: str
    context_ceiling: int | None = None

    def valid(self):
        return (
            _safe_model_id(self.model_id) == self.model_id
            and all(isinstance(value, str) and value.strip() and len(value) <= 1024 for value in (
                self.model_id, self.model_key, self.model_path, self.instance_reference,
                self.generation_marker,
            ))
            and all(type(value) is int for value in (
                self.size_bytes, self.context_tokens, self.output_tokens, self.framing_min,
                self.framing_max,
            ))
            and self.size_bytes > 0
            and 0 < self.output_tokens < self.context_tokens
            and (self.context_ceiling is None or (
                type(self.context_ceiling) is int and self.output_tokens < self.context_ceiling <= self.context_tokens
            ))
            and 0 <= self.framing_min <= self.framing_max < self.context_tokens
            and all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) for value in (
                self.evidence_sha256, self.render_probe_sha256,
            ))
        )

    def identity(self):
        return {
            "identifier": self.model_id, "modelKey": self.model_key,
            "path": self.model_path, "sizeBytes": self.size_bytes,
            "instanceReference": self.instance_reference, "deviceIdentifier": None,
            "format": "gguf",
        }

    def matches(self, metadata):
        return (
            isinstance(metadata, dict)
            and all(key in metadata and type(metadata[key]) is type(value) and metadata[key] == value
                    for key, value in self.identity().items())
        )

    def capacity_matches(self, value):
        return type(value) is int and value == self.context_tokens

    @property
    def admitted_context(self):
        return self.context_ceiling if self.context_ceiling is not None else self.context_tokens


class _Bridge:
    def __init__(self, process):
        self.process = process
        self.replies = queue.Queue(maxsize=2)

    def start_reader(self):
        def read():
            try:
                while line := self.process.stdout.readline(_MAX_FRAME + 1):
                    if len(line) > _MAX_FRAME or not line.endswith(b"\n"):
                        break
                    self.replies.put_nowait(json.loads(line))
                self.process.wait(timeout=2)
            except (OSError, ValueError, queue.Full, subprocess.TimeoutExpired):
                pass
            finally:
                try:
                    self.replies.put_nowait(None)
                except queue.Full:
                    pass

        threading.Thread(target=read, daemon=True, name="lmstudio-bridge-reader").start()

    def _write(self, packet):
        data = json.dumps(packet, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
        if len(data) > _MAX_FRAME:
            raise _fail("input_budget", "LM Studio request exceeds the bridge input limit.")
        self.process.stdin.write(data)
        self.process.stdin.flush()

    def request(self, command, body, cancelled):
        _raise_if_cancelled(cancelled)
        deadline = time.monotonic() + (300 if command == "predict" else 20)
        try:
            self._write({"command": command, **body})
        except OSError:
            if self.process.poll() == 75:
                raise _fail("resource_busy", "The Foreman model slot is occupied or unavailable.") from None
            raise
        while time.monotonic() < deadline:
            _raise_if_cancelled(cancelled)
            try:
                reply = self.replies.get(timeout=0.05)
            except queue.Empty:
                continue
            if reply is None and self.process.poll() == 75:
                raise _fail("resource_busy", "The Foreman model slot is occupied or unavailable.")
            if not isinstance(reply, dict) or reply.get("command") != command:
                break
            if reply.get("code") in {"resource_busy", "incomplete", "invalid_response",
                                      "model_unavailable", "unsupported_tool_output", "cancelled"}:
                raise _fail(reply["code"], "LM Studio rejected the local request.")
            if isinstance(reply.get("value"), dict):
                _raise_if_cancelled(cancelled)
                return reply["value"]
            break
        else:
            raise _fail("timeout", "LM Studio timed out.")
        raise _fail("provider_error", "The optional LM Studio bridge is unavailable.")

    def close(self):
        # EOF asks the bridge to abort. The separate lease owner stays alive
        # while the server may still be predicting, even if this caller exits.
        try:
            self.process.stdin.close()
        except OSError:
            pass


@contextmanager
def _native_session(deployment, cancelled):
    _raise_if_cancelled(cancelled)
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    node, lms = shutil.which("node"), shutil.which("lms")
    foreman = Path("C:/AI/foreman/Claude/runs/local.slots")
    if (os.name != "nt" or not git_bash.is_file() or not node or not lms
            or os.environ.get("LOCAL_SLOTS_DIR") or not foreman.is_dir()):
        raise _fail("provider_error", "The optional LM Studio runtime or shared Foreman slot is unavailable.")
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("NODE_", "BASH_ENV", "ENV", "NPM_", "GIT_CONFIG"))
           and "PROXY" not in key.upper() and not key.upper().endswith(("_KEY", "_TOKEN"))}
    env["MSYS2_ARG_CONV_EXCL"] = "*"
    process = subprocess.Popen(
        [str(git_bash), "--noprofile", "--norc", (_RUNTIME_DIR / "lease.sh").as_posix(),
         foreman.as_posix(), Path(node).as_posix(), (_RUNTIME_DIR / "bridge.mjs").as_posix(),
         Path(lms).as_posix()],
        cwd=_RUNTIME_DIR, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    # The lease shell can outlive this caller while cancellation resolves.
    session = _Bridge(process)
    session.start_reader()
    try:
        yield session
    finally:
        session.close()


class LMStudioRouter(_GenerationRouter):
    provider = "lmstudio"

    def __init__(self, deployment=None, *, session_factory=_native_session):
        super().__init__(profile_rows={})
        self._deployment = deployment
        self._session_factory = session_factory
        self._transaction_lock = threading.Lock()
        self._session = None
        self._prepared = None

    def catalogue(self, refresh=False):
        deployment = self._deployment
        if not isinstance(deployment, Deployment) or not deployment.valid():
            return []
        return [GenerationProfile(
            id=deployment.model_id, provider=self.provider, classes=["chat"],
            quality="local", tier="local", input_usd_per_mtok=0, output_usd_per_mtok=0,
            context_tokens=deployment.admitted_context, output_tokens=deployment.output_tokens,
            priority=0, capabilities=["stream"],
        )]

    def generate(
        self, preferred=None, messages=None, strict=True, stream=True,
        max_output_tokens=1024, cancelled=None, on_chunk=None, *,
        text_only=False, answer_buffer_bytes=None, **options,
    ):
        _raise_if_cancelled(cancelled)
        deployment = self._deployment
        if not self.catalogue() or preferred != deployment.model_id:
            raise _fail("strict_pin", "An eligible, explicitly selected LM Studio deployment is required.")
        if (
            set(options) - {"mode", "provider", "task_class"}
            or options.get("provider", self.provider) != self.provider
            or options.get("mode", "local-chat") not in ("local", "local-chat")
            or options.get("task_class", "chat") != "chat"
            or not stream or not text_only or on_chunk is not None
            or type(answer_buffer_bytes) is not int or not 0 < answer_buffer_bytes <= 65_536
            or type(max_output_tokens) is not int or not 0 < max_output_tokens <= deployment.output_tokens
            or not text_messages(messages or [])
        ):
            raise _fail("invalid_request", "LM Studio requires a pinned, buffered local text request.")
        if not self._transaction_lock.acquire(blocking=False):
            raise _fail("resource_busy", "Another LM Studio request owns the shared model slot.")
        try:
            with self._session_factory(deployment, cancelled) as session:
                self._session = session
                self._cancelled = cancelled
                metadata = session.request("inspect", {
                    "identity": deployment.identity(),
                    "probe": RENDER_PROBE, "probeSha256": deployment.render_probe_sha256,
                    "generationMarker": deployment.generation_marker,
                    "context": deployment.context_tokens, "framingMax": deployment.framing_max,
                    "outputLimit": deployment.output_tokens, "admitContext": deployment.admitted_context,
                }, cancelled)
                if not deployment.matches(metadata):
                    raise _fail("model_unavailable", "LM Studio loaded-model identity changed.")
                if not deployment.capacity_matches(metadata.get("effectiveContext")):
                    raise _fail("input_budget", "The verified LM Studio loaded context is unavailable.")
                return super().generate(
                    preferred=preferred, messages=messages, strict=True, stream=True,
                    max_output_tokens=max_output_tokens, cancelled=cancelled,
                    text_only=True, answer_buffer_bytes=answer_buffer_bytes, **options,
                )
        except GenerationError:
            raise
        except Exception:
            _raise_if_cancelled(cancelled)
            raise _fail("provider_error", "LM Studio local generation failed.") from None
        finally:
            self._session = None
            self._prepared = None
            self._cancelled = None
            self._transaction_lock.release()

    def _admission_inputs(self, profile, messages, options):
        deployment = self._deployment
        prepared = self._session.request("prepare", {"messages": messages}, self._cancelled)
        rendered = prepared.get("rendered")
        count = prepared.get("nativeTokens")
        prompt_sha = prepared.get("promptSha256")
        source = "LM Studio instance-bound effective loaded context"
        if (
            not deployment.capacity_matches(prepared.get("effectiveContext"))
            or not isinstance(rendered, str) or not rendered.endswith(deployment.generation_marker)
            or not preserves_messages(rendered[:-len(deployment.generation_marker)], messages)
            or _sha256(rendered) != prompt_sha
            or type(count) is not int or count <= 0
        ):
            self._prepared = None
            return unknown("LM Studio native rendering or tokenizer parity is unavailable"), None, source
        self._prepared = (copy.deepcopy(messages), prompt_sha, count)
        return TokenCount(
            count + deployment.framing_max,
            "conservative" if deployment.framing_min != deployment.framing_max else "exact",
            f"LM Studio native tokenizer and framing receipt sha256:{deployment.evidence_sha256}",
        ), deployment.admitted_context, source

    def _dispatch(
        self, profile, *, messages, stream, max_output_tokens, options, cancelled,
        on_chunk, text_only, answer_buffer_bytes,
    ):
        deployment = self._deployment
        prepared_messages, prompt_sha, native_count = self._prepared
        if messages != prepared_messages:
            raise _fail("input_budget", "The admitted LM Studio prompt changed.")
        terminal = self._session.request("predict", {
            "promptSha256": prompt_sha, "maxTokens": max_output_tokens,
            "answerBytes": answer_buffer_bytes,
        }, cancelled)
        _raise_if_cancelled(cancelled)
        if not deployment.matches(terminal.get("identity")):
            raise _fail("model_unavailable", "LM Studio terminal model identity changed.")
        stop = terminal.get("stopReason")
        if stop == "toolCalls":
            raise _fail("unsupported_tool_output", "LM Studio emitted an unexpected tool event.")
        if stop != "eosFound" or terminal.get("truncated", False) is not False:
            raise _fail("incomplete", "LM Studio did not complete without truncation.")
        prompt_tokens, output_tokens = terminal.get("promptTokens"), terminal.get("outputTokens")
        answer = terminal.get("text")
        if (
            not deployment.capacity_matches(terminal.get("effectiveContext"))
            or type(terminal.get("maxTokens")) is not int
            or terminal["maxTokens"] != max_output_tokens
            or terminal.get("overflow") != "stopAtLimit"
            or type(prompt_tokens) is not int
            or not native_count + deployment.framing_min <= prompt_tokens <= native_count + deployment.framing_max
            or type(output_tokens) is not int or not 0 < output_tokens <= max_output_tokens
            or not isinstance(answer, str) or not answer.strip()
            or len(answer.encode("utf-8")) > answer_buffer_bytes
        ):
            raise _fail("invalid_response", "LM Studio completion receipts did not match admission.")
        return answer, profile.id, {"input_tokens": prompt_tokens, "output_tokens": output_tokens}, "stop"
