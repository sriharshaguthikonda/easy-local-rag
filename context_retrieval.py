import json

import threading
import ollama

from chroma_hybrid_retrieval import RetrievalError, search_collection, search_keywords
from embedding_contract import EmbeddingContractError
import chromadb
import os


from groq import Groq
from chromadb.config import Settings

from urllib.parse import urljoin
from pathlib import Path

# Set of non-keywords
non_keywords = set(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "for",
        "to",
        "from",
        "up",
        "down",
        "into",
        "over",
        "under",
        "about",
        "between",
        "after",
        "before",
        "while",
        "during",
        "as",
        "but",
        "or",
        "so",
        "such",
        "that",
        "this",
        "these",
        "those",
        "all",
        "any",
        "both",
        "some",
        "most",
        "much",
        "many",
        "few",
        "one",
        "each",
        "every",
        "neither",
        "either",
        "who",
        "whom",
        "whose",
        "what",
        "which",
        "where",
        "when",
        "why",
        "how",
        "it",
        "its",
        "itself",
        "he",
        "she",
        "her",
        "him",
        "his",
        "they",
        "their",
        "them",
        "themselves",
        "you",
        "your",
        "yourself",
        "yourselves",
        "we",
        "our",
        "ours",
        "us",
    ]
)


from dotenv import load_dotenv


groq_model = "llama-3.3-70b-versatile"

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
Groqclient = None
_rewrite_router = None


# Initialize ChromaDB client
collection_name = "html_chunks_temp"
collection = None

# Define the model to use
model = "mxbai-embed-large"


def _result_value(result, name, default=None):
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _router_for_rewrite(router=None):
    global _rewrite_router
    if router is not None:
        return router
    if _rewrite_router is None:
        from generation_models import groq_router

        _rewrite_router = groq_router()
    return _rewrite_router


def _ensure_collection():
    global collection
    if collection is None:
        local_client = chromadb.PersistentClient(
            settings=Settings(anonymized_telemetry=False)
        )
        collection = local_client.get_collection(
            name=collection_name, embedding_function=None
        )
    return collection

def rewrite_input_and_generate_synonyms(original_input, *, router=None):
    """Use the shared Groq boundary and preserve the raw query on failure."""
    system_prompt = (
        "You are a helpful assistant working in a medical context. Your tasks are:\n"
        "1) Rephrase the given input to make it clearer and more precise in one sentence while preserving its original meaning. It will be used for searching ChromaDB after conversion to embeddings.\n"
        "2) Provide a list of synonyms for each keyword in the rephrased input.\n"
        "3) Provide spelling variants (if applicable, such as American and British spellings) for each keyword.\n"
        "4) Provide plural and singular forms for each keyword.\n"
        "5) Provide other parts of speech forms for each keyword.\n"
        "6) Provide closely related terms for each keyword.\n"
        "Respond in the following JSON format:\n"
        "{\n"
        '  "rephrased": "[sentence]",\n'
        '  "keywords": {\n'
        '    "[word]": {\n'
        '      "synonyms": ["synonym1", "synonym2"],\n'
        '      "spelling_variants": ["variant1", "variant2"],\n'
        '      "plural_singular": ["plural_form", "singular_form"],\n'
        '      "parts_of_speech": ["noun form", "verb form"],\n'
        '      "related_terms": ["related1", "related2"]\n'
        "    }\n"
        "  }\n"
        "}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Rewrite and generate synonyms, spelling variants, "
                "plural/singular forms, parts of speech, and related terms "
                f'for: "{original_input}".'
            ),
        },
    ]
    try:
        result = _router_for_rewrite(router).generate(
            preferred=groq_model,
            messages=messages,
            strict=False,
            stream=False,
            max_output_tokens=512,
            response_format={"type": "json_object"},
        )
        raw = _result_value(result, "text", "") or ""
        if isinstance(raw, dict):
            data = raw
        else:
            raw = str(raw).strip()
            if raw.startswith(chr(96) * 3):
                raw = raw.strip(chr(96))
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            data = json.loads(raw)
        rewritten = data.get("rephrased") or data.get("rewritten") or original_input
        keywords = data.get("keywords") or {}
        return rewritten, keywords if isinstance(keywords, dict) else {}
    except Exception as error:
        print(f"Rewrite unavailable; using original query ({type(error).__name__}).")
        return original_input, {}


def print_relevant_context(results):
    print("Context Pulled from Documents:\n")
    for meta in results:
        file_name = meta.get("file_path") or meta.get("file_name") or "Unknown"
        modification_time = meta.get("modification_time", "Unknown")
        text = meta["text"]

        clickable_file_path = urljoin("file:", Path(file_name).absolute().as_uri())

        print(
            f"Text: {text}\n"
            f"File Name: {clickable_file_path}\n"
            f"Modification Time: {modification_time}\n"
        )
        # Open the file in the default web browser
        # webbrowser.open(file_path)


def get_relevant_context_hybrid(
    user_input,
    top_k=5,
    additional_unique_files=5,
    keyword_match=True,
    alpha=0.5,  # Weight for vector similarity
    beta=0.3,  # Weight for keyword match
    gamma=0.2,  # Weight for BM25 score
    lambda_mmr=0.5,  # Balance parameter for MMR
    *,
    rewrite=True,
    router=None,
):
    try:
        if rewrite:
            if router is None:
                rewritten_input, synonym_dict = rewrite_input_and_generate_synonyms(
                    user_input
                )
            else:
                rewritten_input, synonym_dict = rewrite_input_and_generate_synonyms(
                    user_input, router=router
                )
        else:
            rewritten_input, synonym_dict = user_input, {}

        requested_model = model
        words = search_keywords(rewritten_input, synonym_dict) if keyword_match else []
        results = search_collection(
            _ensure_collection(), user_input, embedding_model=requested_model,
            embed=lambda: ollama.embeddings(
                model=requested_model, prompt=rewritten_input, keep_alive=-1,
            )["embedding"],
            top_k=top_k, additional=additional_unique_files, diversity=lambda_mmr,
            vector_weight=alpha, lexical_weight=gamma,
            words=words, keyword_weight=beta if keyword_match else 0,
        )
        sources = [
            {
                **row["meta"], "text": row["document"],
                "collection_id": row["collection_id"], "chunk_id": row["chunk_id"],
            }
            for row in results
        ]

        if results:
            threading.Thread(
                target=print_relevant_context,
                args=(sources,),
                daemon=True,
            ).start()
        context = "\n\n".join(row["document"] for row in results)
        return context

    except (EmbeddingContractError, RetrievalError) as error:
        raise RuntimeError(f"Retrieval failed: {error}") from error
    except Exception as error:
        raise RuntimeError("Retrieval failed; no model request was made") from error
