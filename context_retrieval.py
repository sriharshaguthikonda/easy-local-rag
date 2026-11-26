import numpy as np
import json

import threading
import ollama

from embedding_contract import EmbeddingContractError, validate_embedding_compatibility
import chromadb
import os


from groq import Groq
from chromadb.config import Settings

from rank_bm25 import BM25Okapi
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
        file_name = meta["file_name"]
        modification_time = meta.get("modification_time", "Unknown")
        text = meta["text"]

        clickable_file_path = urljoin("file:", Path(file_name).as_uri())

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
        relevant_context = ""
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

        local_collection = _ensure_collection()

        requested_model = model
        if not validate_embedding_compatibility(local_collection, requested_model):
            return ""

        # Encode the rewritten input into an embedding
        input_embedding = ollama.embeddings(
            model=requested_model,
            prompt=rewritten_input,
            keep_alive=-1,
        )["embedding"]

        # Perform vector similarity search
        search_result = local_collection.query(
            query_embeddings=[input_embedding],
            n_results=50,
            include=["documents", "metadatas", "distances"],
        )

        if not search_result["metadatas"]:
            return "No relevant context found."

        # Extract results with distances
        vector_results = [
            {
                "meta": meta,
                "vector_score": 1.0
                - dist,  # Convert distance to similarity (assuming normalized)
            }
            for meta, dist in zip(
                search_result["metadatas"][0], search_result["distances"][0]
            )
        ]

        # Perform keyword matching if enabled
        keyword_results = []
        if keyword_match:
            # Extract keywords and synonyms
            keywords = [
                word
                for word in rewritten_input.lower().split()
                if word not in non_keywords
            ]

            # Add synonyms to the keyword list
            for key, synonyms in synonym_dict.items():
                keywords.extend(synonyms)

            # Use a set to remove duplicates
            keywords = set(keywords)

            for meta in search_result["metadatas"][0]:
                # Match against both original keywords and their synonyms
                match_score = sum(
                    meta["text"].lower().count(keyword)
                    + meta["file_name"].lower().count(keyword)
                    for keyword in keywords
                )
                if match_score > 0:
                    keyword_results.append({"meta": meta, "keyword_score": match_score})

        # Perform BM25 search
        bm25_corpus = [meta["text"] for meta in search_result["metadatas"][0]]
        bm25 = BM25Okapi([doc.split() for doc in bm25_corpus])
        bm25_scores = bm25.get_scores(rewritten_input.split())

        bm25_results = [
            {"meta": meta, "bm25_score": score}
            for meta, score in zip(search_result["metadatas"][0], bm25_scores)
        ]

        # Normalize scores for vector, keyword, and BM25 results
        max_vector_score = max(
            [res["vector_score"] for res in vector_results], default=1
        )
        max_keyword_score = max(
            [res["keyword_score"] for res in keyword_results], default=1
        )
        max_bm25_score = max([res["bm25_score"] for res in bm25_results], default=1)

        for res in vector_results:
            res["vector_score"] /= max_vector_score

        for res in keyword_results:
            res["keyword_score"] /= max_keyword_score

        for res in bm25_results:
            res["bm25_score"] /= max_bm25_score

        # Combine results using weighted scoring
        combined_results = {}
        for res in vector_results:
            file_name = res["meta"]["file_name"]
            combined_results[file_name] = {
                "meta": res["meta"],
                "final_score": alpha * res["vector_score"],
            }

        for res in keyword_results:
            file_name = res["meta"]["file_name"]
            if file_name in combined_results:
                combined_results[file_name]["final_score"] += (
                    beta * res["keyword_score"]
                )
            else:
                combined_results[file_name] = {
                    "meta": res["meta"],
                    "final_score": beta * res["keyword_score"],
                }

        for res in bm25_results:
            file_name = res["meta"]["file_name"]
            if file_name in combined_results:
                combined_results[file_name]["final_score"] += gamma * res["bm25_score"]
            else:
                combined_results[file_name] = {
                    "meta": res["meta"],
                    "final_score": gamma * res["bm25_score"],
                }

        # Sort by final_score
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["final_score"], reverse=True
        )

        # Limit results to top_k
        final_results = [res["meta"] for res in sorted_results[:top_k]]

        # Use MMR to select additional_unique_files
        remaining_results = [res for res in sorted_results[top_k:]]
        selected_additional_files = []

        for res in remaining_results:
            max_similarity = max(
                [
                    np.dot(res["meta"]["embedding"], selected["meta"]["embedding"])
                    for selected in selected_additional_files
                ],
                default=0,
            )
            mmr_score = (
                lambda_mmr * res["final_score"] - (1 - lambda_mmr) * max_similarity
            )
            res["mmr_score"] = mmr_score

        # Sort by MMR score to get the most relevant and diverse results
        sorted_additional_files = sorted(
            remaining_results, key=lambda x: x["mmr_score"], reverse=True
        )

        selected_additional_files = sorted_additional_files[:additional_unique_files]

        # Combine the top_k and the selected additional unique files
        final_results.extend([res["meta"] for res in selected_additional_files])

        # Prepare relevant context
        relevant_context = "\n\n".join([res["text"] for res in final_results])

        # Start a worker thread to print details of the results
        worker_thread = threading.Thread(
            target=print_relevant_context,
            args=(final_results,),
            daemon=True,
        )
        worker_thread.start()

        return relevant_context

    except Exception as e:
        if isinstance(e, EmbeddingContractError):
            raise RuntimeError(f"Retrieval failed: {e}") from e
        raise RuntimeError("Retrieval failed; no model request was made") from e
