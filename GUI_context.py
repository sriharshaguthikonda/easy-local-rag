import os
import sys
import re
import json
import subprocess

from groq import Groq
from chroma_hybrid_retrieval import CAPTURE_PROCESS_TIMEOUT


def rewrite_input_and_generate_synonyms(settings, user_input, *, on_result=None):
    """Rewrite a query through the selected provider's shared router.

    Local generation intentionally keeps the user's exact query: a Groq
    rewrite would make local retrieval provider-dependent.  Every online
    rewrite goes through the shared provider router.
    """
    provider = settings.get("generation_provider", "groq")
    if provider == "ollama":
        return user_input, {}

    router = settings.get("generation_router_groq") or settings.get("generation_router")
    if router is None and provider == "groq":
        try:
            import generation_models

            router = generation_models.groq_router(client=settings.get("groq_client"))
        except Exception as error:
            print(f"[Rewrite] Router unavailable: {getattr(error, 'code', 'generation_failed')}")
            return user_input, {}

    system_prompt = (
        "You are a helpful assistant. Your tasks are:\n"
        "1) Rephrase the given input to make it clearer in one sentence.\n"
        "2) Provide synonyms, spelling variants, plural/singular forms for keywords.\n"
        "Respond in JSON format:\n"
        '{"rephrased": "[sentence]", "keywords": {"[word]": {"synonyms": [], "spelling_variants": [], "plural_singular": [], "parts_of_speech": [], "related_terms": []}}}'
    )

    if router is not None:
        try:
            result = router.generate(
                preferred=settings.get("groq_rewrite_model"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f'Rewrite and generate synonyms for: "{user_input}"',
                    },
                ],
                strict=bool(settings.get("strict_generation_model", False)),
                stream=False,
                max_output_tokens=int(settings.get("rewrite_max_output_tokens", 512)),
                cancelled=lambda: False,
                response_format={"type": "json_object"},
                task_class="chat",
            )
            response_json = getattr(result, "text", result)
            response_data = json.loads(str(response_json).strip())
            if on_result is not None:
                on_result(result)
            return response_data.get("rephrased", user_input), response_data.get("keywords", {})
        except Exception as error:
            print(
                "[Rewrite] Router error: %s"
                % getattr(error, "code", "generation_failed")
            )
            return user_input, {}

    return user_input, {}


def get_relevant_context_hybrid(settings, user_input):
    """Retrieve evidence in the isolated Chroma process used by Qt."""
    if settings.get("generation_provider") == "ollama":
        rewritten, synonyms = user_input, {}
    else:
        rewritten, synonyms = rewrite_input_and_generate_synonyms(settings, user_input)

    request = {
        "chromadb_path": settings["chromadb_path"],
        "collection_name": settings["collection_name"],
        "collection_id": settings.get("collection_id"),
        "embedding_model": settings["embedding_model"],
        "query": user_input,
        "rewritten": rewritten,
        "synonyms": synonyms,
        "n_results": int(settings["top_k"]),
        "vector_weight": float(settings["alpha"]),
        "keyword_weight": float(settings["beta"]),
        "lexical_weight": float(settings["gamma"]),
        "hybrid": True,
    }
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GUI_chromadb_helper.py")
    result = subprocess.run(
        [sys.executable, helper], input=json.dumps(request), capture_output=True,
        encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=CAPTURE_PROCESS_TIMEOUT,
    )
    if result.returncode:
        raise RuntimeError(f"ChromaDB helper failed with status {result.returncode}")
    output = json.loads(result.stdout)
    if not output["success"]:
        raise RuntimeError(f"ChromaDB query failed: {output['error']}")
    return output["results"]
