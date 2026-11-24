"""Safe, provider-scoped model discovery and generation fallback.

This module deliberately keeps the provider SDKs optional at import time.  The
application can therefore use its offline/search-only paths without importing a
client or reading a credential.  A router owns one authenticated client/session
and its in-memory catalogue; catalogues are never shared between accounts.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse


CATALOGUE_TTL_SECONDS = 300.0
DISCOVERY_TIMEOUT_SECONDS = 10.0
MAX_GENERATION_ATTEMPTS = 3
_CAPABILITIES = frozenset(
    {"stream", "tools", "json_object", "json_schema", "reasoning_off"}
)
_CLASSES = frozenset({"chat", "reasoning"})
_MODEL_UNAVAILABLE_CODES = frozenset(
    {
        "model_not_found",
        "model_not_available",
        "model_not_found_error",
        "model_unavailable",
        "invalid_model",
        "model_decommissioned",
    }
)


class GenerationError(RuntimeError):
    """A safe, typed generation failure.

    ``message`` is intentionally a short category-level explanation.  Provider
    bodies and exception strings are not retained because they may contain API
    keys, URLs, account identifiers, or prompts.
    """

    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        preferred_model: str | None = None,
        actual_model: str | None = None,
        provider: str | None = None,
        attempts: Sequence[str] | None = None,
    ) -> None:
        normalized = re.sub(r"[^a-z0-9_.-]+", "_", str(code).lower()).strip("_")
        self.code = normalized or "generation_error"
        self.message = _safe_message(message or self.code)
        self.preferred_model = _safe_model_id(preferred_model)
        self.actual_model = _safe_model_id(actual_model)
        self.provider = provider if provider in {"groq", "ollama"} else None
        self.attempts = tuple(
            item for item in (_safe_model_id(value) for value in (attempts or ())) if item
        )
        super().__init__(self.message)


def _safe_message(value: Any) -> str:
    """Return a bounded message without accepting provider exception text."""

    text = str(value)
    # Keep diagnostics category-like and avoid accidentally exposing a URL or a
    # bearer token if a caller supplied a raw exception by mistake.
    text = re.sub(r"(?i)bearer\s+[a-z0-9._-]+", "bearer [redacted]", text)
    text = re.sub(r"https?://\S+", "[redacted-url]", text)
    text = re.sub(r"[A-Za-z0-9_-]{32,}", "[redacted]", text)
    return text[:240]


def _safe_model_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if "://" in text or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,159}", text):
        return None
    return text


@dataclass
class GenerationProfile:
    """Normalized, provider-reported model profile."""

    id: str
    provider: str
    classes: list[str]
    quality: str
    tier: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    context_tokens: int
    output_tokens: int
    priority: int
    capabilities: list[str]
    profiled: bool = True
    available: bool = True
    remote: bool = False
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def operating_mode(self) -> str:
        return "local" if self.provider == "ollama" else "cloud"


@dataclass
class GenerationResult:
    text: str
    preferred_model: str | None
    actual_model: str | None
    reason: str = ""
    provider: str | None = None
    attempted_models: list[str] = field(default_factory=list)
    usage: Any = None


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    try:
        return getattr(obj, key)
    except Exception:
        return default


def _mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    for method_name in ("model_dump", "dict"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                result = method()
            except Exception:
                continue
            if isinstance(result, Mapping):
                return result
    return {}


def _as_sequence(value: Any) -> list[Any]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return []
    if isinstance(value, Sequence):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result


def _is_local_host(host: str | None) -> bool:
    if not host:
        return True
    raw = str(host).strip()
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    return (
        parsed.scheme in {"http", "https"}
        and parsed.username is None
        and hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
    )


def _is_cloud_model(model: str) -> bool:
    lower = model.lower().strip()
    return (
        lower.endswith(":cloud")
        or lower.endswith("-cloud")
        or lower.endswith("/cloud")
        or lower.startswith("cloud/")
    )


def _contains_remote(value: Any) -> bool:
    """Inspect provider metadata for explicit remote/cloud manifest markers."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower().replace("-", "_")
            if lowered in {"remote_model", "remote_host", "remote_endpoint", "cloud_model", "cloud_host"}:
                if item not in (None, "", False, 0):
                    return True
            if lowered in {"remote", "cloud"} and isinstance(item, bool) and item:
                return True
            if lowered in {"remote", "cloud"} and isinstance(item, str) and item:
                return True
        return any(_contains_remote(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_remote(item) for item in value)
    for key in ("remote_model", "remote_host", "remote_endpoint", "cloud_model", "cloud_host"):
        marker = _value(value, key)
        if marker not in (None, "", False, 0):
            return True
    return False


def _normalise_caps(values: Any) -> set[str]:
    out: set[str] = set()
    for value in _as_sequence(values):
        lowered = str(value).lower().strip().replace("-", "_")
        if lowered in {"completion", "generate", "text_generation", "chat", "generate_content"}:
            continue
        if lowered in {"stream", "streaming"}:
            out.add("stream")
        elif lowered in {"tool", "tools", "function_calling", "function_call"}:
            out.add("tools")
        elif lowered in {"json", "json_object", "structured_output"}:
            out.add("json_object")
        elif lowered in {"response_format", "structured_outputs"}:
            out.add("json_object")
        elif lowered in {"json_schema", "structured_json", "schema"}:
            out.add("json_schema")
        elif lowered in {"reasoning_off", "no_reasoning"}:
            out.add("reasoning_off")
        elif lowered in {"embedding", "embeddings", "speech", "tts", "audio"}:
            out.add(lowered)
    return out


def _model_id(row: Any) -> str:
    return str(_value(row, "id", _value(row, "model", _value(row, "name", ""))) or "").strip()


def _context_from_metadata(*values: Any) -> int | None:
    keys = {
        "context_length",
        "context_window",
        "num_ctx",
        "max_context_length",
        "general.context_length",
        "llama.context_length",
    }
    limits = []
    for value in values:
        for metadata in (value, _value(value, "model_info"), _value(value, "modelinfo")):
            reported = [_as_int(_value(metadata, key)) for key in keys]
            reported.extend(
                _as_int(item) for key, item in _mapping(metadata).items()
                if str(key).lower().replace("/", ".").endswith(".context_length")
            )
            limits.extend(limit for limit in reported if limit > 0)
    return min(limits, default=None)


def _configured_profiles(provider: str) -> dict[str, dict[str, Any]]:
    path_value = os.environ.get("EASY_RAG_GENERATION_PROFILES")
    if path_value:
        try:
            path = Path(path_value)
            raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(path_value)
        except (OSError, ValueError, TypeError) as exc:
            raise GenerationError("configuration", "generation profile configuration is invalid") from exc
    else:
        path = Path(__file__).with_name("config") / "generation-models.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise GenerationError("configuration", "generation profile configuration is unavailable") from exc
    if not isinstance(raw, Mapping) or raw.get("version") != 1:
        raise GenerationError("configuration", "generation profile schema is unsupported")
    providers = raw.get("providers")
    if not isinstance(providers, Mapping) or not isinstance(providers.get(provider), Mapping):
        raise GenerationError("configuration", "provider profile configuration is missing")
    rows: dict[str, dict[str, Any]] = {}
    expected_tier = "on_demand" if provider == "groq" else "local"
    for model, row in providers[provider].items():
        if _safe_model_id(model) != model or not isinstance(row, Mapping):
            continue
        classes = row.get("classes")
        capabilities = row.get("capabilities")
        quality = row.get("quality")
        prices = [row.get("input_usd_per_mtok"), row.get("output_usd_per_mtok")]
        sizes = [row.get("context_tokens"), row.get("output_tokens")]
        if (
            not isinstance(classes, list)
            or not classes
            or not all(isinstance(item, str) and item in _CLASSES for item in classes)
            or not isinstance(capabilities, list)
            or not all(isinstance(item, str) and item in _CAPABILITIES for item in capabilities)
            or not isinstance(quality, str)
            or not quality.strip()
            or row.get("tier") != expected_tier
            or not all(type(price) in (int, float) and 0 <= price < float("inf") for price in prices)
            or not all(type(size) is int and size > 0 for size in sizes)
            or type(row.get("priority")) is not int
        ):
            continue
        rows[model] = {**row, "quality": quality.strip()}
    return rows


def _profile_limit(configured, reported, default: int | None) -> int | None:
    limits = (_as_int(configured), _as_int(reported))
    return min((limit for limit in limits if limit > 0), default=default)


def _profile_from_config(
    provider: str,
    model: str,
    config: Mapping[str, Any] | None,
    *,
    metadata: Mapping[str, Any] | None = None,
    remote: bool = False,
    profiled: bool = True,
    classes: Sequence[str] | None = None,
    capabilities: Iterable[str] | None = None,
    context_tokens: int | None = None,
    output_tokens: int | None = None,
    status: str = "active",
) -> GenerationProfile:
    row = config or {}
    row_classes = list(classes or _as_sequence(row.get("classes")))
    row_caps = list(capabilities if capabilities is not None else row.get("capabilities", ["stream"]))
    if not row_classes:
        row_classes = ["chat"]
    return GenerationProfile(
        id=model,
        provider=provider,
        classes=[str(value) for value in row_classes],
        quality=str(row.get("quality", "unknown")),
        tier=str(row.get("tier", "local" if provider == "ollama" else "on_demand")),
        input_usd_per_mtok=row.get("input_usd_per_mtok", 0.0),
        output_usd_per_mtok=row.get("output_usd_per_mtok", 0.0),
        context_tokens=_profile_limit(row.get("context_tokens"), context_tokens, 4096),
        output_tokens=_profile_limit(row.get("output_tokens"), output_tokens, 1024),
        priority=_as_int(row.get("priority"), 1000),
        capabilities=sorted(set(str(value) for value in row_caps if str(value) in _CAPABILITIES)),
        profiled=profiled,
        available=True,
        remote=remote,
        status=status,
        metadata=dict(metadata or {}),
    )


def _catalogue_sort(profiles: Iterable[GenerationProfile]) -> list[GenerationProfile]:
    return sorted(profiles, key=lambda row: (row.priority, row.id))


def _cancel_requested(cancelled: Any) -> bool:
    if cancelled is None:
        return False
    if callable(cancelled):
        try:
            return bool(cancelled())
        except Exception:
            return True
    is_set = getattr(cancelled, "is_set", None)
    if callable(is_set):
        try:
            return bool(is_set())
        except Exception:
            return True
    return bool(cancelled)


def _raise_if_cancelled(cancelled: Any) -> None:
    if _cancel_requested(cancelled):
        raise GenerationError("cancelled", "generation was cancelled")


def _extract_error_status(exc: BaseException) -> int | None:
    for name in ("status_code", "status"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _extract_error_body(exc: BaseException) -> Any:
    body = getattr(exc, "body", None)
    if body is not None:
        return body
    body = getattr(exc, "error", None)
    if body is not None:
        return body
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return response.json()
        except (ValueError, AttributeError):
            pass
    return None


def _error_code(body: Any) -> str:
    if isinstance(body, Mapping):
        error = body.get("error", body)
        if isinstance(error, Mapping):
            for key in ("code", "type", "error_code"):
                value = error.get(key)
                if value:
                    return str(value).lower().strip()
            message = error.get("message")
            if message:
                return str(message).lower().strip()
        if isinstance(error, str):
            return error.lower().strip()
    return str(body or "").lower().strip()


def _classify_provider_error(exc: BaseException, provider: str, model: str) -> str:
    """Map an SDK exception to a safe retry category."""

    status = _extract_error_status(exc)
    body_code = _error_code(_extract_error_body(exc))
    name = type(exc).__name__.lower()
    if status in {401, 403} or "authentication" in name or "invalid_api_key" in body_code:
        return "authentication"
    if provider == "groq":
        # A Groq 404 is retryable only when the structured API body explicitly
        # identifies the missing model.  Other 404s are terminal.
        if status == 404 and body_code in _MODEL_UNAVAILABLE_CODES:
            return "model_unavailable"
        return "provider_error"
    # Ollama returns a model-specific JSON error without a separate code.
    if status in {400, 404}:
        candidate_names = {model.lower()}
        if ":" not in model.rsplit("/", 1)[-1]:
            candidate_names.add(f"{model.lower()}:latest")
        for candidate_name in candidate_names:
            if any(body_code.startswith(f"model {quote}{candidate_name}{quote} not found")
                   for quote in ("'", '"')):
                return "model_unavailable"
    return "provider_error"


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        if content is not None:
            return _extract_text(content)
        text = value.get("text")
        if text is not None:
            return _extract_text(text)
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "".join(_extract_text(item) for item in value)
    content = _value(value, "content")
    if content is not None:
        return _extract_text(content)
    text = _value(value, "text")
    if text is not None:
        return _extract_text(text)
    return ""


def _extract_delta(chunk: Any) -> tuple[str, str | None]:
    model = _safe_model_id(_value(chunk, "model"))
    choices = _value(chunk, "choices")
    if choices:
        first = _as_sequence(choices)[0]
        delta = _value(first, "delta")
        if delta is not None:
            return _extract_text(delta), model
        message = _value(first, "message")
        if message is not None:
            return _extract_text(message), model
        return _extract_text(_value(first, "text")), model
    message = _value(chunk, "message")
    if message is not None:
        return _extract_text(message), model
    return _extract_text(_value(chunk, "response")), model


def _extract_response_text(response: Any) -> tuple[str, str | None, Any]:
    text, model = _extract_delta(response)
    usage = _value(response, "usage")
    if text:
        return text, model, usage
    return _extract_text(response), model, usage


def _is_embedding_id(model: str) -> bool:
    lower = model.lower()
    return any(token in lower for token in ("embed", "minilm", "bge-", "e5-", "rerank"))


def _is_non_generation_id(model: str) -> bool:
    lower = model.lower()
    return _is_embedding_id(lower) or any(
        token in lower for token in ("whisper", "guard", "moderation", "tts", "speech")
    )


def _estimate_tokens(messages, options) -> int:
    """Bound request size without assuming an unfamiliar model's tokenizer."""
    payload = list(messages) + [
        options[key] for key in ("tools", "response_format", "format") if key in options
    ]
    size = len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
    return size + 12 * len(messages) + 32


def _response_format_kind(value: Any) -> str | None:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"json", "json_object"}:
            return "json_object"
        if lowered in {"json_schema", "schema"}:
            return "json_schema"
        return None
    if isinstance(value, Mapping):
        kind = str(value.get("type", "")).lower()
        if kind in {"json", "json_object"}:
            return "json_object"
        if kind in {"json_schema", "schema"}:
            return "json_schema"
        # Ollama accepts a schema itself as ``format``.
        if "properties" in value or "$schema" in value:
            return "json_schema"
    return None


def _policy_options(
    provider_options: Mapping[str, Any],
    *,
    stream: bool,
) -> dict[str, Any]:
    options = dict(provider_options)
    task_class = str(
        options.get("task_class", options.get("task", options.get("request_class", "chat")))
        or "chat"
    ).lower()
    mode = str(options.get("mode", "") or "").lower()
    reasoning = options.get("reasoning")
    if reasoning is True or options.get("reasoning_effort") not in (None, "", "none", "off", False):
        task_class = "reasoning"
    if mode in {"reasoning", "think", "thinking"}:
        task_class = "reasoning"
    elif mode in {"answer", "chat", "plain"}:
        task_class = "chat"
    options["_task_class"] = task_class
    options["_reasoning_disabled"] = reasoning is False or options.get("reasoning_off") is True
    required = set(str(value) for value in _as_sequence(options.get("required_capabilities", [])))
    if stream:
        required.add("stream")
    if _as_sequence(options.get("tools")):
        required.add("tools")
    format_kind = _response_format_kind(options.get("response_format", options.get("format")))
    if format_kind:
        required.add(format_kind)
    if options.get("json_schema") is not None:
        required.add("json_schema")
    if options.get("json_object"):
        required.add("json_object")
    options["_required_capabilities"] = required
    options["_stream"] = bool(stream)
    return options


def _budget_value(options: Mapping[str, Any], name: str) -> float | None:
    direct = options.get(name)
    if direct is None:
        direct = options.get(f"max_{name}")
    budget = options.get("budget")
    if direct is None and isinstance(budget, Mapping):
        direct = budget.get(name, budget.get(f"max_{name}"))
    if direct is None:
        return None
    try:
        value = float(direct)
    except (TypeError, ValueError):
        return -1.0
    return value if value >= 0 else -1.0


def _ollama_context(options):
    request_options = options.get("options")
    if not isinstance(request_options, Mapping):
        return 4096
    return _as_int(request_options.get("num_ctx", 4096))


class _GenerationRouter:
    """Base implementation shared by the two single-provider routers."""

    provider: str

    def __init__(
        self,
        client: Any = None,
        *,
        host: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._client = client
        self._host = host
        self._clock = clock or time.monotonic
        self._catalogue: list[GenerationProfile] | None = None
        self._catalogue_at = float("-inf")
        self._catalogue_lock = threading.RLock()
        self._profile_rows = _configured_profiles(self.provider)

    @property
    def client(self) -> Any:
        return self._ensure_client()

    def _ensure_client(self) -> Any:
        raise NotImplementedError

    def _discover(self, client: Any) -> list[GenerationProfile]:
        raise NotImplementedError

    def _request(self, profile, messages, stream, max_output_tokens, options):
        raise NotImplementedError

    def catalogue(self, refresh: bool = False) -> list[GenerationProfile]:
        """Return a normalized, TTL-cached catalogue for this client/session."""

        now = self._clock()
        with self._catalogue_lock:
            if (
                self._catalogue is not None
                and not refresh
                and now >= self._catalogue_at
                and now - self._catalogue_at < CATALOGUE_TTL_SECONDS
            ):
                return copy.deepcopy(self._catalogue)
            try:
                profiles = _catalogue_sort(self._discover(self._ensure_client()))
            except GenerationError:
                raise
            except Exception:
                raise GenerationError("catalogue_error", "Provider catalogue is unavailable.", provider=self.provider) from None
            self._catalogue = profiles
            self._catalogue_at = now
            return copy.deepcopy(profiles)

    def _eligible(
        self,
        profiles: Sequence[GenerationProfile],
        *,
        preferred: str | None,
        messages: Sequence[Any],
        max_output_tokens: int,
        options: Mapping[str, Any],
        preferred_profile: GenerationProfile | None,
    ) -> list[GenerationProfile]:
        task_class = str(options.get("_task_class", "chat"))
        required = set(options.get("_required_capabilities", set()))
        if (
            self.provider == "groq"
            and "json_schema" in required
            and (options.get("_stream") or "tools" in required)
        ):
            return []
        quality = options.get("quality")
        tier = options.get("tier")
        if tier is not None:
            tier = str(tier)
        if self.provider == "groq" and tier in {"flex", "performance", "cloud"}:
            return []
        input_tokens = max(
            _as_int(options.get("input_tokens")),
            _as_int(options.get("max_input_tokens")),
            _estimate_tokens(messages, options),
        )
        explicit_context = _as_int(options.get("context_tokens"), 0)
        if self.provider == "ollama":
            ollama_context = _ollama_context(options)
            if ollama_context <= 0:
                return []
            explicit_context = min(explicit_context or ollama_context, ollama_context)
        input_budget = _budget_value(options, "input_usd_per_mtok")
        output_budget = _budget_value(options, "output_usd_per_mtok")
        if input_budget is not None and input_budget < 0:
            return []
        if output_budget is not None and output_budget < 0:
            return []
        selected: list[GenerationProfile] = []
        for profile in profiles:
            if not profile.available or profile.remote or profile.provider != self.provider:
                continue
            is_preferred = preferred is not None and profile.id == preferred
            # Unknown local IDs are selectable explicitly but can never become
            # automatic fallback candidates because their spend/capability
            # ceiling is not verified.
            if not profile.profiled and not (
                is_preferred
                and self.provider == "ollama"
                and bool(profile.metadata.get("show_verified"))
            ):
                continue
            if task_class not in profile.classes:
                continue
            if quality is not None and str(quality) != profile.quality:
                continue
            if preferred_profile is not None and profile.quality != preferred_profile.quality:
                continue
            if tier is not None and tier != profile.tier:
                continue
            if options.get("_reasoning_disabled") and "reasoning_off" not in profile.capabilities:
                continue
            if not required.issubset(set(profile.capabilities)):
                continue
            if profile.output_tokens < max_output_tokens:
                continue
            effective_context = profile.context_tokens
            if explicit_context:
                effective_context = min(effective_context, explicit_context)
            if input_tokens + max_output_tokens > effective_context:
                continue
            if input_budget is not None and profile.input_usd_per_mtok > input_budget:
                continue
            if output_budget is not None and profile.output_usd_per_mtok > output_budget:
                continue
            if preferred_profile is not None:
                if profile.input_usd_per_mtok > preferred_profile.input_usd_per_mtok:
                    continue
                if profile.output_usd_per_mtok > preferred_profile.output_usd_per_mtok:
                    continue
                if profile.tier != preferred_profile.tier:
                    continue
            selected.append(profile)
        return _catalogue_sort(selected)

    def _close_stream(self, stream: Any) -> None:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _consume_stream(
        self,
        stream: Any,
        *,
        model_id: str,
        cancelled: Any,
        on_chunk: Callable[[str], Any] | None,
    ) -> tuple[str, str | None, Any]:
        chunks: list[str] = []
        tool_delta_seen = False
        actual_model: str | None = None
        usage: Any = None
        try:
            for chunk in stream:
                _raise_if_cancelled(cancelled)
                piece, model = _extract_delta(chunk)
                choices = _value(chunk, "choices") or []
                message = _value(choices[0], "delta") if choices else _value(chunk, "message")
                tool_delta_seen |= bool(_value(message, "tool_calls") or _value(message, "function_call"))
                actual_model = model or actual_model
                usage = _value(chunk, "usage", usage)
                if piece:
                    chunks.append(piece)
                    if on_chunk is not None:
                        try:
                            on_chunk(piece)
                        except Exception as exc:
                            raise GenerationError("stream_error", "generation stream callback failed") from exc
                    _raise_if_cancelled(cancelled)
            return "".join(chunks), actual_model, usage
        except GenerationError:
            raise
        except Exception as exc:
            category = _classify_provider_error(exc, self.provider, model_id)
            if (chunks or tool_delta_seen) and category == "model_unavailable":
                category = "stream_error"
            raise GenerationError(category, "Generation stream failed.", provider=self.provider) from None
        finally:
            # Closing a completed generator is harmless and releases HTTP
            # resources promptly on both SDKs.
            self._close_stream(stream)

    def _dispatch(
        self,
        profile: GenerationProfile,
        *,
        messages: Sequence[Any],
        stream: bool,
        max_output_tokens: int,
        options: Mapping[str, Any],
        cancelled: Any,
        on_chunk: Callable[[str], Any] | None,
    ) -> tuple[str, str | None, Any]:
        _raise_if_cancelled(cancelled)
        try:
            response = self._request(
                profile, copy.deepcopy(list(messages)), stream, max_output_tokens, options
            )
        except GenerationError:
            raise
        except Exception as error:
            category = _classify_provider_error(error, self.provider, profile.id)
            raise GenerationError(category, "Model request failed.", provider=self.provider) from None
        if stream:
            return self._consume_stream(
                response, model_id=profile.id, cancelled=cancelled, on_chunk=on_chunk
            )
        _raise_if_cancelled(cancelled)
        text, actual_model, usage = _extract_response_text(response)
        if text and on_chunk is not None:
            try:
                on_chunk(text)
            except Exception:
                raise GenerationError("stream_error", "Generation output callback failed.") from None
        return text, actual_model, usage

    def generate(
        self,
        preferred: str | None = None,
        messages: Sequence[Any] | None = None,
        strict: bool = False,
        stream: bool = True,
        max_output_tokens: int = 1024,
        cancelled: Any = None,
        on_chunk: Callable[[str], Any] | None = None,
        **provider_options: Any,
    ) -> GenerationResult:
        """Generate with bounded same-provider/model-class fallback.

        The prepared message sequence is copied once and reused for every
        candidate.  A retry is permitted only for a typed, pre-output model
        unavailability error; all other failures are terminal.
        """

        _raise_if_cancelled(cancelled)
        if max_output_tokens is None:
            max_output_tokens = 1024
        try:
            max_output_tokens = int(max_output_tokens)
        except (TypeError, ValueError) as exc:
            raise GenerationError("invalid_request", "max_output_tokens must be positive") from exc
        if max_output_tokens <= 0:
            raise GenerationError("invalid_request", "max_output_tokens must be positive")
        prepared_messages = list(messages or [])
        options = _policy_options(provider_options, stream=bool(stream))
        requested_provider = provider_options.get("provider")
        if requested_provider is not None and str(requested_provider).lower() != self.provider:
            raise GenerationError("provider_policy", "provider switch is not permitted", provider=self.provider)
        requested_mode = str(provider_options.get("mode", "") or "").lower()
        expected_mode = "local" if self.provider == "ollama" else "cloud"
        if requested_mode in {"offline", "offline-search", "local", "local-chat", "cloud", "cloud-chat"}:
            if requested_mode not in {expected_mode, f"{expected_mode}-chat"}:
                raise GenerationError("provider_policy", "provider is unavailable in the selected mode", provider=self.provider)
        if self.provider == "groq" and provider_options.get("n") not in (None, 1):
            raise GenerationError("invalid_request", "generation requires one response per request", provider=self.provider)
        requested_service_tier = provider_options.get("service_tier")
        if self.provider == "groq" and requested_service_tier not in (None, "on_demand"):
            raise GenerationError("tier_policy", "only the approved on-demand tier is permitted", provider=self.provider)
        preferred_id = _safe_model_id(preferred)
        if preferred_id is None and (preferred or strict):
            raise GenerationError(
                "strict_pin" if strict else "invalid_request",
                "a valid preferred generation model is required",
                provider=self.provider,
            )
        profiles = self.catalogue()
        _raise_if_cancelled(cancelled)
        preferred_profile = next((row for row in profiles if row.id == preferred_id), None) if preferred_id else None
        policy_profiles = profiles
        stale_preference = False
        if preferred_id and preferred_profile is None:
            # A saved, known profile may be stale.  Try it once so the provider
            # can return its typed model-unavailable response, then refresh the
            # same account catalogue.  Unknown IDs and strict pins fail closed.
            static = self._profile_rows.get(preferred_id)
            if strict or static is None:
                raise GenerationError(
                    "strict_pin" if strict else "no_candidate",
                    "preferred generation model is unavailable",
                    preferred_model=preferred_id,
                    provider=self.provider,
                )
            preferred_profile = _profile_from_config(
                self.provider,
                preferred_id,
                static,
                metadata={"stale_preference": True},
                profiled=True,
                status="configured",
            )
            if self.provider == "groq":
                # Groq can prove a retired saved ID by returning its typed 404;
                # keep it as the first safe attempt and refresh afterwards.
                policy_profiles = [*profiles, preferred_profile]
            else:
                # Ollama's list/show is the installed-model boundary.  An ID
                # absent from that verified list must never be dispatched.
                policy_profiles = profiles
                stale_preference = True
        eligible = self._eligible(
            policy_profiles,
            preferred=preferred_id,
            messages=prepared_messages,
            max_output_tokens=max_output_tokens,
            options=options,
            preferred_profile=preferred_profile,
        )
        if (
            preferred_profile is not None
            and preferred_profile.id not in {row.id for row in eligible}
            and not stale_preference
        ):
            raise GenerationError(
                "strict_pin" if strict else "no_candidate",
                "preferred generation model does not satisfy request policy",
                preferred_model=preferred_profile.id,
                provider=self.provider,
            )
        if preferred_id:
            candidate = next((row for row in eligible if row.id == preferred_id), None)
            if candidate is None:
                if not eligible:
                    raise GenerationError(
                        "no_candidate",
                        "no eligible generation model",
                        preferred_model=preferred_id,
                        provider=self.provider,
                    )
                candidate = eligible[0]
        elif eligible:
            candidate = eligible[0]
            preferred_id = candidate.id
            preferred_profile = candidate
        else:
            raise GenerationError("no_candidate", "no eligible generation model", provider=self.provider)

        attempted: list[str] = []
        failed: set[str] = set()
        fallback_reason = (
            f"fallback from {preferred_id} to {candidate.id}: saved model unavailable"
            if stale_preference else ""
        )
        for _attempt in range(MAX_GENERATION_ATTEMPTS):
            _raise_if_cancelled(cancelled)
            if candidate.id in failed:
                break
            attempted.append(candidate.id)
            try:
                text, actual_model, usage = self._dispatch(
                    candidate,
                    messages=prepared_messages,
                    stream=bool(stream),
                    max_output_tokens=max_output_tokens,
                    options=options,
                    cancelled=cancelled,
                    on_chunk=on_chunk,
                )
                actual_id = _safe_model_id(actual_model) or candidate.id
                reason = fallback_reason
                if actual_id != candidate.id:
                    alias = f"provider reported {actual_id} for {candidate.id}"
                    reason = f"{reason}. {alias}" if reason else alias
                return GenerationResult(
                    text=text,
                    preferred_model=preferred_id,
                    actual_model=actual_id,
                    reason=reason,
                    provider=self.provider,
                    attempted_models=list(attempted),
                    usage=usage,
                )
            except GenerationError as exc:
                if exc.code == "cancelled":
                    raise
                if exc.code != "model_unavailable":
                    # Keep the safe category and model context without
                    # exposing the provider's raw response.
                    raise GenerationError(
                        exc.code,
                        exc.message,
                        preferred_model=preferred_id,
                        actual_model=candidate.id,
                        provider=self.provider,
                        attempts=attempted,
                    ) from exc
                failed.add(candidate.id)
                with self._catalogue_lock:
                    self._catalogue = None
                if strict or len(attempted) >= MAX_GENERATION_ATTEMPTS:
                    raise GenerationError(
                        "model_unavailable",
                        "preferred generation model is unavailable",
                        preferred_model=preferred_id,
                        actual_model=candidate.id,
                        provider=self.provider,
                        attempts=attempted,
                    ) from exc
                # A typed unavailable-model error invalidates this router's
                # catalogue.  Refresh through the same client/session, then
                # recompute policy and budgets before selecting a replacement.
                _raise_if_cancelled(cancelled)
                profiles = self.catalogue(refresh=True)
                _raise_if_cancelled(cancelled)
                eligible = self._eligible(
                    profiles,
                    preferred=preferred_id,
                    messages=prepared_messages,
                    max_output_tokens=max_output_tokens,
                    options=options,
                    preferred_profile=preferred_profile,
                )
                replacements = [row for row in eligible if row.id not in failed]
                if not replacements:
                    raise GenerationError(
                        "model_unavailable",
                        "no replacement generation model is available",
                        preferred_model=preferred_id,
                        actual_model=candidate.id,
                        provider=self.provider,
                        attempts=attempted,
                    ) from exc
                candidate = replacements[0]
                fallback_reason = f"fallback from {preferred_id} to {candidate.id} after model_unavailable"
        raise GenerationError(
            "model_unavailable",
            "no replacement generation model is available",
            preferred_model=preferred_id,
            provider=self.provider,
            attempts=attempted,
        )


class GroqRouter(_GenerationRouter):
    provider = "groq"

    def __init__(self, client=None, *, clock=None):
        if client is not None and callable(getattr(client, "with_options", None)):
            client = client.with_options(timeout=DISCOVERY_TIMEOUT_SECONDS, max_retries=0)
        super().__init__(client, clock=clock)

    def _ensure_client(self):
        if self._client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise GenerationError("catalogue_error", "Groq client is not configured", provider=self.provider)
            from groq import Groq

            self._client = Groq(
                api_key=api_key, timeout=DISCOVERY_TIMEOUT_SECONDS, max_retries=0
            )
        return self._client

    def _request(self, profile, messages, stream, max_output_tokens, options):
        client = self._ensure_client()
        completions = _value(_value(client, "chat"), "completions")
        create = _value(completions, "create")
        if not callable(create):
            raise GenerationError("provider_error", "Groq generation is unavailable", provider=self.provider)
        allowed = {
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
            "reasoning_effort",
            "reasoning_format",
            "include_reasoning",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "parallel_tool_calls",
            "user",
            "metadata",
            "n",
        }
        kwargs = {key: value for key, value in options.items() if key in allowed}
        kwargs.update(
            {
                "model": profile.id,
                "messages": messages,
                "stream": bool(stream),
                "max_completion_tokens": max_output_tokens,
                "service_tier": "on_demand",
            }
        )
        if profile.id.startswith("openai/gpt-oss") and options.get("_task_class") == "chat":
            kwargs.setdefault("include_reasoning", False)
            kwargs.setdefault("reasoning_effort", "low")
        # Never allow a caller's lower-level max token key to weaken the
        # boundary; max_completion_tokens is the authoritative ceiling.
        kwargs.pop("max_tokens", None)
        return create(**kwargs)

    def _discover(self, client: Any) -> list[GenerationProfile]:
        models = _value(_value(client, "models"), "list")
        if not callable(models):
            raise GenerationError("catalogue_error", "Groq model listing is unavailable", provider=self.provider)
        response = models()
        raw_rows = _value(response, "data", response if isinstance(response, Sequence) else [])
        rows = _as_sequence(raw_rows)
        profiles: list[GenerationProfile] = []
        for row in rows:
            model = _model_id(row)
            if not model:
                continue
            if _is_non_generation_id(model):
                continue
            active = _value(row, "active", True)
            if active is False or str(active).lower() in {"false", "inactive", "disabled", "0"}:
                continue
            config = self._profile_rows.get(model)
            metadata = {
                "active": bool(active),
                "owned_by": str(_value(row, "owned_by", ""))[:120],
            }
            supported_value = _value(row, "supported_parameters", None)
            supported = supported_value if supported_value is not None else []
            raw_row_caps = {str(value).lower() for value in _as_sequence(_value(row, "capabilities", []))}
            if raw_row_caps.intersection({"embedding", "embeddings", "speech", "tts", "audio", "moderation"}):
                continue
            metadata["supported_parameters"] = [str(value)[:80] for value in _as_sequence(supported)[:40]]
            caps = set(_as_sequence(config.get("capabilities") if config else []))
            if config is None:
                caps.update(_normalise_caps(supported))
            if supported_value is not None and isinstance(supported, Sequence) and not isinstance(supported, (str, bytes, bytearray)):
                supported_lower = {str(value).lower() for value in supported}
                if "tools" not in supported_lower and "tool" not in supported_lower and "function_call" not in supported_lower:
                    caps.discard("tools")
                if not supported_lower.intersection({"response_format", "json", "json_object", "json_schema", "structured_outputs"}):
                    caps.discard("json_object")
                    caps.discard("json_schema")
            if config is None and not caps:
                caps.add("stream")
            profiles.append(
                _profile_from_config(
                    self.provider,
                    model,
                    config,
                    metadata=metadata,
                    profiled=config is not None,
                    capabilities=caps,
                    context_tokens=_context_from_metadata(row),
                    output_tokens=_profile_limit(
                        _value(row, "max_output_tokens"), _value(row, "max_completion_tokens"), None
                    ),
                    status="active" if active is not False else "inactive",
                )
            )
        return profiles


class OllamaRouter(_GenerationRouter):
    provider = "ollama"

    def __init__(
        self,
        client: Any = None,
        *,
        host: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(client, host=host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434", clock=clock)

    def _ensure_client(self) -> Any:
        if not _is_local_host(self._host):
            raise GenerationError("remote_provider", "Ollama requires a local endpoint", provider=self.provider)
        if self._client is not None:
            http_client = getattr(self._client, "_client", None)
            endpoint = getattr(http_client, "base_url", None)
            if endpoint is not None and not _is_local_host(str(endpoint)):
                raise GenerationError("remote_provider", "Ollama requires a local endpoint", provider=self.provider)
            if getattr(http_client, "follow_redirects", False) or getattr(http_client, "trust_env", False):
                raise GenerationError("remote_provider", "Ollama transport must disable redirects and environment proxies", provider=self.provider)
            return self._client
        try:
            import ollama

            self._client = ollama.Client(
                host=self._host,
                timeout=DISCOVERY_TIMEOUT_SECONDS,
                follow_redirects=False,
                trust_env=False,
            )
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError("catalogue_error", "Ollama client is unavailable", provider=self.provider) from exc
        return self._client

    @staticmethod
    def _metadata(client, model=None):
        # ollama-python's typed responses can discard cloud manifest fields.
        # Use the same client transport and inspect the server JSON directly.
        transport = getattr(client, "_client", None)
        if transport is not None and getattr(transport, "base_url", None) is not None:
            if model is None:
                response = transport.get("/api/tags")
            else:
                response = transport.post("/api/show", json={"model": model})
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError("Invalid Ollama catalogue response")
            return payload
        return client.list() if model is None else client.show(model)

    def _request(self, profile, messages, stream, max_output_tokens, options):
        client = self._ensure_client()
        if not _is_local_host(self._host):
            raise GenerationError("remote_provider", "Ollama requires a local endpoint", provider=self.provider)
        transport = getattr(client, "_client", None)
        if transport is not None and getattr(transport, "base_url", None) is not None:
            manifest = self._metadata(client, profile.id)
            if not manifest or _contains_remote(manifest):
                raise GenerationError("remote_provider", "Ollama model must be verified as local", provider=self.provider)
        chat = _value(client, "chat")
        if not callable(chat):
            raise GenerationError("provider_error", "Ollama generation is unavailable", provider=self.provider)
        kwargs: dict[str, Any] = {
            "model": profile.id,
            "messages": messages,
            "stream": bool(stream),
        }
        if "tools" in options:
            kwargs["tools"] = options["tools"]
        format_value = options.get("format", options.get("response_format"))
        format_kind = _response_format_kind(format_value)
        if format_kind == "json_object":
            kwargs["format"] = "json"
        elif format_kind == "json_schema":
            if isinstance(format_value, Mapping) and format_value.get("type") == "json_schema":
                kwargs["format"] = format_value.get("json_schema", format_value)
            else:
                kwargs["format"] = format_value
        explicit_options = options.get("options")
        ollama_options = dict(explicit_options) if isinstance(explicit_options, Mapping) else {}
        requested_context = min(_ollama_context(options), profile.context_tokens)
        ollama_options["num_ctx"] = requested_context
        ollama_options["num_predict"] = min(max_output_tokens, profile.output_tokens)
        # Common Ollama generation knobs live inside ``options`` rather than
        # Groq's top-level request.  Preserve only documented knobs.
        for key in ("temperature", "top_p", "top_k", "seed", "repeat_penalty", "stop"):
            if key in options:
                ollama_options[key] = options[key]
        kwargs["options"] = ollama_options
        if "keep_alive" in options:
            kwargs["keep_alive"] = options["keep_alive"]
        if "think" in options:
            kwargs["think"] = options["think"]
        elif options.get("_task_class") == "reasoning":
            kwargs["think"] = True
        elif options.get("_reasoning_disabled"):
            kwargs["think"] = False
        return chat(**kwargs)

    def _discover(self, client: Any) -> list[GenerationProfile]:
        response = self._metadata(client)
        raw_rows = _value(response, "models", response if isinstance(response, Sequence) else [])
        rows = _as_sequence(raw_rows)
        profiles: list[GenerationProfile] = []
        for row in rows:
            model = _model_id(row)
            if not model or _is_embedding_id(model):
                continue
            remote = _is_cloud_model(model) or _contains_remote(_mapping(row))
            show_data: Any = None
            show_verified = False
            try:
                show_data = self._metadata(client, model)
                show_verified = bool(_mapping(show_data))
            except Exception:
                # Unverified IDs are displayed but remain ineligible.
                show_data = None
            remote = remote or _contains_remote(_mapping(show_data))
            show_caps = _normalise_caps(
                _value(show_data, "capabilities", _value(row, "capabilities", []))
            )
            raw_caps = {str(value).lower() for value in _as_sequence(_value(show_data, "capabilities", []))}
            if raw_caps and raw_caps.issubset({"embedding", "embeddings", "speech", "tts", "audio"}):
                continue
            config = self._profile_rows.get(model)
            context = _context_from_metadata(show_data, row)
            if config is None:
                # Unknown local IDs are visible for explicit user selection,
                # but they are never automatic fallback candidates.
                classes = ["chat"]
                caps = set(show_caps)
                caps.add("stream")
                profile = _profile_from_config(
                    self.provider,
                    model,
                    None,
                    metadata={
                        "remote": remote,
                        "show_capabilities": sorted(show_caps),
                        "show_verified": show_verified,
                    },
                    remote=remote,
                    profiled=False,
                    classes=classes,
                    capabilities=caps,
                    context_tokens=context or 4096,
                    output_tokens=1024,
                )
            else:
                caps = set(_as_sequence(config.get("capabilities")))
                # Server metadata may narrow optional capabilities.  Ollama
                # always supports streaming and JSON-object formatting; tools
                # and JSON schema require explicit metadata/configuration.
                if raw_caps:
                    if "tools" not in show_caps:
                        caps.discard("tools")
                    if "json_schema" not in show_caps:
                        # Keep a configured schema capability when the server
                        # gives no claim, but remove it when a capability list
                        # explicitly lacks structured output.
                        if "json_schema" not in raw_caps and "json" not in raw_caps:
                            caps.discard("json_schema")
                    if "json_object" not in show_caps and "json" not in raw_caps:
                        caps.discard("json_object")
                profile = _profile_from_config(
                    self.provider,
                    model,
                    config,
                    metadata={
                        "remote": remote,
                        "show_capabilities": sorted(show_caps),
                        "show_verified": show_verified,
                    },
                    remote=remote,
                    profiled=show_verified,
                    capabilities=caps,
                    context_tokens=context,
                )
            profiles.append(profile)
        return profiles


def groq_router(
    client: Any = None,
    *,
    clock: Callable[[], float] | None = None,
) -> GroqRouter:
    """Create a lazy, account-scoped Groq router."""

    return GroqRouter(client=client, clock=clock)


def ollama_router(
    client: Any = None,
    host: str | None = None,
    *,
    clock: Callable[[], float] | None = None,
) -> OllamaRouter:
    """Create a lazy, local-endpoint Ollama router."""

    return OllamaRouter(client=client, host=host, clock=clock)


__all__ = [
    "CATALOGUE_TTL_SECONDS",
    "DISCOVERY_TIMEOUT_SECONDS",
    "GenerationError",
    "GenerationProfile",
    "GenerationResult",
    "GroqRouter",
    "OllamaRouter",
    "groq_router",
    "ollama_router",
]
