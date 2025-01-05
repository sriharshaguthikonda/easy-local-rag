import numpy as np
from rank_bm25 import BM25Okapi
from urllib.parse import urljoin
from pathlib import Path
import threading

# ...existing code...


def rewrite_input_and_generate_synonyms(original_input):
    # ...existing code...
    pass


def print_relevant_context(results):
    print("Context Pulled from Documents:\n")
    for meta in results:
        file_name = meta["file_name"]
        modification_time = meta.get("modification_time", "Unknown")
        text = meta["text"]

        clickable_file_path = urljoin("file:", Path(file_name).as_uri())

        print(
            f"{YELLOW}Text:{RESET_COLOR} {text}\n"
            f"{BLUE}File Name:{clickable_file_path}\n{RESET_COLOR}"
            f"{PINK}Modification Time: {modification_time}\n{RESET_COLOR}"
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
):
    try:
        relevant_context = ""
        rewritten_input, synonym_dict = rewrite_input_and_generate_synonyms(user_input)

        # Encode the rewritten input into an embedding
        input_embedding = ollama.embeddings(
            model=model,
            prompt=rewritten_input,
            keep_alive=-1,
        )["embedding"]

        # Perform vector similarity search
        search_result = collection.query(
            query_embeddings=[input_embedding],
            n_results=50,
            include=["documents", "metadatas", "distances"],
        )

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
        print(f"An error occurred: {e}")
        return "Answer this yourself!"
