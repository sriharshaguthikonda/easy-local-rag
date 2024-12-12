from typing import List, Dict
from groq import Groq

groq = Groq()


def get_synonyms_for_keywords(keywords: List[str]) -> Dict[str, List[str]]:
    """
    Fetch synonyms for given keywords using Groq.
    """
    try:
        keyword_query = ", ".join(keywords)
        chat_completion = groq.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a linguistic model that provides synonyms for given keywords in JSON format. "
                    "Output a dictionary where each key is a keyword, and the value is a list of its synonyms.",
                },
                {
                    "role": "user",
                    "content": f"Provide synonyms for the following keywords: {keyword_query}",
                },
            ],
            model="llama3-8b-8192",
            temperature=0,
            stream=False,
            response_format={"type": "json_object"},
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"An error occurred while fetching synonyms: {e}")
        return {}


def apply_mmr(query_embedding, results, top_k):
    """
    Apply Maximal Marginal Relevance (MMR) to select diverse results.
    """
    selected_results = []
    remaining_results = results.copy()

    while len(selected_results) < top_k and remaining_results:
        best_result = max(
            remaining_results,
            key=lambda res: calculate_mmr_score(query_embedding, res, selected_results),
        )
        selected_results.append(best_result)
        remaining_results.remove(best_result)

    return selected_results


def calculate_mmr_score(query_embedding, result, selected_results):
    """
    Compute MMR score based on relevance and diversity.
    """
    relevance = cosine_similarity(query_embedding, result["embedding"])
    diversity = (
        max(
            cosine_similarity(result["embedding"], sel["embedding"])
            for sel in selected_results
        )
        if selected_results
        else 0
    )
    lambda_value = 0.5
    return lambda_value * relevance - (1 - lambda_value) * diversity


from ollama import embeddings


def hybrid_search_with_synonyms(
    rewritten_input: str,
    collection,
    model,
    top_k=5,
    additional_unique_files=10,
    keyword_match=True,
):
    """
    Perform a hybrid search combining semantic similarity and keyword-based search with synonym expansion.
    """
    try:
        keywords = rewritten_input.lower().split()
        synonyms_dict = get_synonyms_for_keywords(keywords)
        expanded_keywords = keywords + [
            syn for syn_list in synonyms_dict.values() for syn in syn_list
        ]

        input_embedding = embeddings(
            model=model,
            prompt=rewritten_input,
            keep_alive=-1,
        )["embedding"]

        search_result = collection.query(
            query_embeddings=[input_embedding],
            n_results=50,
            include=["documents", "metadatas", "distances"],
        )

        keyword_results = []
        if keyword_match:
            keyword_results = [
                meta
                for meta in search_result["metadatas"][0]
                if any(
                    keyword in meta["text"].lower()
                    or keyword in meta["file_name"].lower()
                    for keyword in expanded_keywords
                )
            ]

        unique_results = {
            meta["file_name"]: meta
            for meta in (search_result["metadatas"][0] + keyword_results)
        }
        combined_results = list(unique_results.values())

        additional_files = apply_mmr(
            input_embedding, combined_results, top_k=additional_unique_files
        )

        final_results = combined_results[:top_k] + additional_files
        relevant_context = "\n\n".join([meta["text"] for meta in final_results])
        return relevant_context

    except Exception as e:
        print(f"An error occurred during hybrid search: {e}")
        return "Answer this yourself!"
