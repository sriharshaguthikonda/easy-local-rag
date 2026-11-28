"""Shared synthetic provider fixtures."""

import json

import pytest


@pytest.fixture
def synthetic_provider_admission(monkeypatch):
    """Give legacy transport/validator fixtures a known synthetic capacity."""
    import generation_models as gm
    from token_budget import TokenCount

    def measured(self, profile, messages, options):
        context = profile.context_tokens
        if context is None:
            context = 131072
        if self.provider == "ollama":
            requested = gm._ollama_context(options)
            if requested is not None:
                context = min(context, requested)
        count = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        return TokenCount(count, "conservative", "synthetic test deployment"), context, "synthetic metadata"

    for router in (gm.GroqRouter, gm.OllamaRouter):
        monkeypatch.setattr(router, "_admission_inputs", measured)
