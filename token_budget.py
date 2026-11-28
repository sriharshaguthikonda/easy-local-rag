"""Offline token counts and the shared history/admission data contract."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class TokenCount:
    tokens: int | None
    quality: Literal["exact", "conservative", "unknown"]
    provenance: str


@dataclass(frozen=True, slots=True)
class Admission:
    input_count: TokenCount
    context_tokens: int | None
    output_tokens: int
    context_source: str
    history_turns_removed: int = 0


def unknown(reason: str) -> TokenCount:
    return TokenCount(None, "unknown", reason)


def text_messages(messages: Sequence) -> bool:
    return bool(messages) and all(
        isinstance(message, Mapping)
        and message.keys() == {"role", "content"}
        and message["role"] in ("system", "user", "assistant")
        and isinstance(message["content"], str)
        for message in messages
    )


def oldest_turn_removed(messages: list) -> list | None:
    """Keep the protected prefix and current question byte for byte."""
    if not text_messages(messages) or messages[-1]["role"] != "user":
        return None
    start = next((i for i, message in enumerate(messages) if message["role"] != "system"), 0)
    if start + 2 >= len(messages):
        return None
    if (messages[start]["role"], messages[start + 1]["role"]) != ("user", "assistant"):
        return None
    return messages[:start] + messages[start + 2:]


def preserves_messages(rendered: str, messages: Sequence) -> bool:
    """Check the rendered prompt contains each original text in order."""
    offset = 0
    for message in messages:
        content = message["content"]
        index = rendered.find(content, offset)
        if index < 0:
            return False
        offset = index + len(content)
    return True


def _verified_asset(path, digest, limit: int) -> bytes:
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Missing asset binding")
    asset_path = Path(path)
    if not asset_path.is_absolute() or asset_path.drive.startswith(("\\\\", "//")):
        raise ValueError("Use a local absolute asset path")
    asset_path = asset_path.resolve(strict=True)
    if asset_path.drive.startswith(("\\\\", "//")):
        raise ValueError("Use a local absolute asset path")
    with asset_path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit or sha256(data).hexdigest() != digest:
        raise ValueError("Asset digest mismatch")
    return data


def render_hosted_chat(spec: Mapping, messages: Sequence, request_values: Mapping) -> str:
    """Render a hash-pinned deployment template without filesystem loaders."""
    from jinja2 import StrictUndefined
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    template = _verified_asset(spec.get("template_path"), spec.get("template_sha256"), 262144)
    environment = ImmutableSandboxedEnvironment(undefined=StrictUndefined)
    variables = spec.get("template_variables", {})
    if not isinstance(variables, dict) or {"messages", "add_generation_prompt"} & variables.keys():
        raise ValueError("Invalid chat template configuration")
    return environment.from_string(template.decode("utf-8")).render(
        messages=messages, add_generation_prompt=True, **(variables | request_values),
    )


def count_rendered(
    spec: Mapping, rendered: str, *, provider: str, model: str, template: str,
) -> TokenCount:
    """Count a completely rendered, already verified model request."""
    revision = spec.get("revision")
    extra = spec.get("max_added_tokens")
    add_special = spec.get("add_special_tokens")
    if (
        spec.get("provider") != provider or spec.get("model") != model
        or not isinstance(revision, str) or not revision.strip()
        or type(extra) is not int or extra < 0 or type(add_special) is not bool
    ):
        return unknown("Unverified tokenizer or framing binding")
    try:
        from tokenizers import Tokenizer

        data = _verified_asset(spec.get("tokenizer_path"), spec.get("tokenizer_sha256"), 67108864)
        rendered.encode("utf-8")
        tokenizer = Tokenizer.from_str(data.decode("utf-8"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        count = len(tokenizer.encode(rendered, add_special_tokens=add_special).ids)
    except Exception:
        # The Rust-backed JSON parser uses plain Exception for invalid assets.
        return unknown("Tokenizer asset unavailable or incompatible")
    provenance = (
        f"{provider}/{model}@{revision} "
        f"tokenizer sha256:{spec['tokenizer_sha256']} template {template}"
    )
    return TokenCount(count + extra, "conservative" if extra else "exact", provenance)
