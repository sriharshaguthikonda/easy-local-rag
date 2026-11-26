"""Hybrid rank fusion for vector + lexical retrieval legs."""


def fuse_candidates(vector, lexical, *, vector_weight, lexical_weight):
    """Fuse vector and lexical candidate lists into a single ranked list.

    Each leg is iterated in order (vector first, then lexical). Within a leg,
    rows are keyed by ``(collection_id, chunk_id)``; only the first occurrence
    of each key is kept and every unique row increments that leg's rank. The
    contribution of a row to its ``final_score`` is ``weight / (60 + rank)``.

    A row copied from the vector leg keeps all of its fields. When a lexical
    row overlaps an already-copied key, its ``bm25_score`` is copied over if
    present and its contribution is added to the running ``final_score``.

    Returns the fused rows sorted by ``(-final_score, collection_id, chunk_id)``.
    """
    results = {}

    for leg, weight in ((vector, vector_weight), (lexical, lexical_weight)):
        if weight <= 0:
            continue

        seen = set()
        rank = 0
        for row in leg:
            key = (row["collection_id"], row["chunk_id"])
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            contribution = weight / (60 + rank)

            merged = results.get(key)
            if merged is None:
                new_row = dict(row)
                new_row["final_score"] = contribution
                results[key] = new_row
            else:
                merged["final_score"] = merged.get("final_score", 0.0) + contribution
                if "bm25_score" in row:
                    merged["bm25_score"] = row["bm25_score"]

    fused = list(results.values())
    fused.sort(key=lambda r: (-r["final_score"], r["collection_id"], r["chunk_id"]))
    return fused
