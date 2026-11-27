"""Synthetic provider completions for maintained answer-path regressions."""

import json

from generation_models import GenerationResult


def excerpt_payload(quote, alias=1):
    return json.dumps({"excerpts": [{"alias": alias, "quote": quote}]})


def completed_excerpt(quote, *, actual_model="selected", preferred_model=None, reason=""):
    return GenerationResult(
        excerpt_payload(quote), preferred_model or actual_model, actual_model,
        reason=reason, finish_reason="stop",
    )


def displayed_excerpt(quote, alias=1):
    return f"Structurally validated excerpts\n“{quote}” [{alias}]"
