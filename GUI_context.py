import os
import sys
import re
import json
import subprocess

import ollama
from groq import Groq
from rank_bm25 import BM25Okapi


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
    """Get relevant context using hybrid search - runs in main thread"""
    print("[Context] Getting rewritten input...")
    if settings.get("generation_provider") == "ollama":
        # Do not perform a paid Groq rewrite for a local generation request.
        rewritten_input, synonym_dict = user_input, {}
    else:
        rewritten_input, synonym_dict = rewrite_input_and_generate_synonyms(settings, user_input)

    print(f"[Context] Getting embeddings with model: {settings['embedding_model']}")
    input_embedding = ollama.embeddings(
        model=settings['embedding_model'],
        prompt=rewritten_input,
        keep_alive=-1,
    )["embedding"]
    print(f"[Context] Got embedding, length: {len(input_embedding)}")

    print("[Context] Querying ChromaDB via subprocess...")
    helper_script = os.path.join(
        os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv and sys.argv[0] else 'rag_gui.py')),
        'GUI_chromadb_helper.py',
    )
    python_exe = sys.executable

    input_data = json.dumps({
        'chromadb_path': settings['chromadb_path'],
        'collection_name': settings['collection_name'],
        'embedding': input_embedding,
        'n_results': 20,
    })

    result = subprocess.run(
        [python_exe, helper_script],
        input=input_data,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise Exception(f"ChromaDB helper failed: {result.stderr}")

    output = json.loads(result.stdout)
    if not output.get('success'):
        raise Exception(f"ChromaDB query failed: {output.get('error')}")

    search_result = {
        'documents': [output['documents']],
        'metadatas': [output['metadatas']],
        'distances': [output['distances']],
    }
    print(f"[Context] Got {len(output['documents'])} results")

    # Vector results
    vector_results = [
        {"meta": meta, "document": doc, "vector_score": 1.0 - dist}
        for meta, doc, dist in zip(
            search_result["metadatas"][0],
            search_result["documents"][0],
            search_result["distances"][0],
        )
    ]

    # Keyword matching
    non_keywords = {"is", "a", "an", "and", "the", "of", "in", "on", "at", "by", "with", "for", "to", "from"}
    keywords = [word for word in rewritten_input.lower().split() if word not in non_keywords]

    for key, details in synonym_dict.items():
        keywords.append(key)
        for field in ['synonyms', 'spelling_variants', 'plural_singular', 'parts_of_speech', 'related_terms']:
            if details.get(field):
                keywords.extend(details[field])

    keywords = list(set(keywords))

    keyword_results = []
    for meta, doc in zip(search_result["metadatas"][0], search_result["documents"][0]):
        normalized_doc = re.sub(r"(?<=\w)-\s*(?=\w)", "", doc.lower())
        match_score = sum(len(re.findall(rf"\b{re.escape(kw)}\b", normalized_doc)) for kw in keywords)
        if match_score > 0:
            keyword_results.append({"meta": meta, "document": doc, "keyword_score": match_score})

    # BM25
    bm25_corpus = search_result["documents"][0]
    bm25 = BM25Okapi([doc.split() for doc in bm25_corpus])
    bm25_scores = bm25.get_scores(rewritten_input.split())

    bm25_results = [
        {"meta": meta, "document": doc, "bm25_score": score}
        for meta, doc, score in zip(
            search_result["metadatas"][0],
            search_result["documents"][0],
            bm25_scores,
        )
    ]

    # Normalize and combine
    alpha, beta, gamma = settings['alpha'], settings['beta'], settings['gamma']
    max_vector = max([r["vector_score"] for r in vector_results], default=1)
    max_keyword = max([r["keyword_score"] for r in keyword_results], default=1)
    max_bm25 = max([r["bm25_score"] for r in bm25_results], default=1)

    combined = {}
    for res in vector_results:
        fn = res["meta"].get("file_name", "unknown")
        combined[fn] = {
            "meta": res["meta"],
            "document": res["document"],
            "final_score": alpha * (res["vector_score"] / max_vector),
            "keywords": keywords,
        }

    for res in keyword_results:
        fn = res["meta"].get("file_name", "unknown")
        if fn in combined:
            combined[fn]["final_score"] += beta * (res["keyword_score"] / max_keyword)
        else:
            combined[fn] = {
                "meta": res["meta"],
                "document": res["document"],
                "final_score": beta * (res["keyword_score"] / max_keyword),
                "keywords": keywords,
            }

    for res in bm25_results:
        fn = res["meta"].get("file_name", "unknown")
        if fn in combined:
            combined[fn]["final_score"] += gamma * (res["bm25_score"] / max_bm25)
        else:
            combined[fn] = {
                "meta": res["meta"],
                "document": res["document"],
                "final_score": gamma * (res["bm25_score"] / max_bm25),
                "keywords": keywords,
            }

    sorted_results = sorted(combined.values(), key=lambda x: x["final_score"], reverse=True)
    print(f"[Context] Returning top {settings['top_k']} results")
    return sorted_results[:settings['top_k']]
