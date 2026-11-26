"""Independent lexical and vector reads of a consistent Chroma collection."""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from chromadb.api.models.Collection import Collection
import numpy as np

from embedding_contract import EmbeddingContractError, validate_embedding_compatibility
from hybrid_rank_fusion import fuse_candidates


PAGE_SIZE = 512
MAX_CHUNK_BYTES = 512 * 1024
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
MAX_INDEX_BYTES = 2 * 1024**3
CAPTURE_SECONDS = 90
LOCK_SECONDS = 15
CAPTURE_PROCESS_TIMEOUT = CAPTURE_SECONDS + 2 * LOCK_SECONDS + 10


class RetrievalError(RuntimeError):
    """A complete retrieval could not be produced."""


def canonical_store_path(path):
    return os.path.normcase(str(Path(path).resolve()))


@contextmanager
def store_lock(path):
    """Serialize native readers and managed writers for the same store."""
    state_root = Path(
        os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME")
        or (Path.home() / ".local" / "state")
    )
    lock_dir = state_root / "easy-local-rag" / "chroma-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(canonical_store_path(path).encode("utf-8")).hexdigest()
    with (lock_dir / f"{digest}.lock").open("a+b", buffering=0) as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            def acquire():
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

        deadline = time.monotonic() + LOCK_SECONDS
        while True:
            try:
                acquire()
                break
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise RetrievalError("Timed out waiting for Chroma ingestion or retrieval.") from error
                time.sleep(0.05)
        yield


def persistent_selection(collection):
    """Return the connection coordinates for a managed native collection."""
    if not isinstance(collection, Collection):
        return None
    client = collection._client
    settings = client.get_settings()
    if not settings.is_persistent:
        return None
    tenant = collection.tenant
    database = collection.database
    return {
        "path": canonical_store_path(settings.persist_directory),
        "name": collection.name,
        "tenant": tenant,
        "database": database,
        "expected_id": str(collection.id),
    }


def open_client(path, *, tenant=None, database=None):
    import chromadb
    from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings

    return chromadb.PersistentClient(
        path=path, settings=Settings(anonymized_telemetry=False),
        tenant=tenant or DEFAULT_TENANT, database=database or DEFAULT_DATABASE,
    )


def _open_selection(selection):
    if not (Path(selection["path"]) / "chroma.sqlite3").is_file():
        raise RetrievalError("The local Chroma database is missing.")
    return open_client(
        selection["path"], tenant=selection.get("tenant"), database=selection.get("database"),
    ).get_collection(selection["name"], embedding_function=None)


def _check_selection(collection, expected_id, embedding_model):
    if expected_id is not None and str(collection.id) != expected_id:
        raise RetrievalError("The selected Chroma collection changed during retrieval.")
    return embedding_model is None or validate_embedding_compatibility(collection, embedding_model)


def _inspect(selection, embedding_model):
    with store_lock(selection["path"]):
        collection = _open_selection(selection)
        if not _check_selection(collection, selection.get("expected_id"), embedding_model):
            return None
        if collection.count() == 0:
            return None
        return str(collection.id)


def _capture_selection(selection, embedding_model, options):
    with store_lock(selection["path"]):
        collection = _open_selection(selection)
        if (
            not _check_selection(collection, selection.get("expected_id"), embedding_model)
            or collection.count() == 0
        ):
            return []
        return capture(collection, **options)


def _run_operation(operation, selection, **arguments):
    payload = json.dumps({"operation": operation, "selection": selection, **arguments})
    try:
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            input=payload, capture_output=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=CAPTURE_PROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as error:
        raise RetrievalError("The Chroma operation exceeded its time limit.") from error
    if child.returncode:
        raise RetrievalError(f"The Chroma operation process exited with status {child.returncode}.")
    try:
        response = json.loads(child.stdout)
    except ValueError as error:
        raise RetrievalError("The Chroma operation returned an invalid response.") from error
    if "error" in response:
        error_type = EmbeddingContractError if response.get("contract") else RetrievalError
        raise error_type(response["error"])
    return response["result"]


def search_path_in_worker(path, name, query, *, embedding_model=None, embed=None, expected_id=None, **options):
    """Search from a fresh operation process that has never opened the vector index."""
    vector_enabled = options.get("vector_weight", 0) > 0
    if vector_enabled and embedding_model is None:
        raise EmbeddingContractError("A valid embedding_model is required.")
    selection = {"path": canonical_store_path(path), "name": name, "expected_id": expected_id}
    collection_id = _inspect(selection, embedding_model)
    if collection_id is None:
        return []
    selection["expected_id"] = collection_id
    options["embedding"] = embed() if vector_enabled else None
    return _capture_selection(selection, embedding_model, {"query": query, **options})


def self_test_query(path, name, embedding_model, expected_id, embed, top_k, include):
    if embedding_model is None:
        raise EmbeddingContractError("A valid embedding_model is required.")
    selection = {"path": canonical_store_path(path), "name": name, "expected_id": expected_id}
    collection_id = _inspect(selection, embedding_model)
    if collection_id is None:
        empty = dict.fromkeys(["documents", "metadatas", "distances", "embeddings", "uris", "data"])
        empty.update({field: [[]] for field in include if field in empty})
        return {**empty, "ids": [[]], "included": include}
    embedding = embed()
    with store_lock(selection["path"]):
        collection = _open_selection(selection)
        if _check_selection(collection, collection_id, embedding_model):
            result = collection.query(query_embeddings=[embedding], n_results=top_k, include=include)
            return {**result, "collection_id": collection_id}
        raise EmbeddingContractError("The collection lost its embedding_model during retrieval.")


def search_collection(collection, query, *, embedding_model=None, embed=None, **options):
    """Use a fresh native index for persistent Chroma and support injected collections."""
    selection = persistent_selection(collection)
    vector_enabled = options.get("vector_weight", 0) > 0
    if vector_enabled and embedding_model is None:
        raise EmbeddingContractError("A valid embedding_model is required.")
    if selection:
        return _search_persistent(selection, query, embedding_model, embed, options)
    if not _check_selection(collection, None, embedding_model) or collection.count() == 0:
        return []
    options["embedding"] = embed() if vector_enabled else None
    if not _check_selection(collection, None, embedding_model):
        return []
    return capture(collection, query, **options)


def search_store(path, name, query, *, embedding_model=None, embed=None, expected_id=None, **options):
    """Read a persisted collection from a long-lived caller without opening a native index there."""
    selection = {"path": canonical_store_path(path), "name": name, "expected_id": expected_id}
    return _search_persistent(selection, query, embedding_model, embed, options)


def _search_persistent(selection, query, embedding_model, embed, options):
    embedding = None
    if options.get("vector_weight", 0) > 0:
        if embedding_model is None:
            raise EmbeddingContractError("A valid embedding_model is required.")
        current_id = _run_operation("inspect", selection, embedding_model=embedding_model)
        if current_id is None:
            return []
        selection = {**selection, "expected_id": current_id}
        embedding = embed()
    return _run_operation(
        "capture", selection, embedding_model=embedding_model,
        options={**options, "query": query, "embedding": embedding},
    )


def distance_key(value, *, descending=False):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return (False, -value if descending else value)
    return (True, 0)


def matches_filter(metadata, document, filter_text):
    filename = str((metadata or {}).get("file_name") or "")
    return filter_text in f"{filename} {document or ''}".casefold()


def _fts_query(text):
    terms = sorted(set(re.findall(r"[^\W_]+", text.lower())))
    if len(terms) > 256 or sum(map(len, terms)) > 8192:
        raise RetrievalError("The lexical search exceeds the query term limit.")
    return " OR ".join(f'"{term}"' for term in terms)


@contextmanager
def _lexical_index(enabled):
    if not enabled:
        yield None
        return
    try:
        with TemporaryDirectory(prefix="easy-rag-bm25-") as directory:
            connection = sqlite3.connect(Path(directory) / "lexical.sqlite")
            try:
                connection.execute("PRAGMA cache_size = -4096")
                connection.execute("PRAGMA temp_store = FILE")
                connection.execute("PRAGMA journal_mode = OFF")
                page_size = connection.execute("PRAGMA page_size").fetchone()[0]
                connection.execute(f"PRAGMA max_page_count = {MAX_INDEX_BYTES // page_size}")
                connection.execute("CREATE TABLE seen (chunk_id TEXT PRIMARY KEY) WITHOUT ROWID")
                connection.execute(
                    "CREATE VIRTUAL TABLE evidence USING fts5(chunk_id UNINDEXED, document, tokenize='unicode61')"
                )
                yield connection
            finally:
                connection.close()
    except sqlite3.Error as error:
        raise RetrievalError(f"Could not build or search the bounded SQLite FTS5 index: {error}") from error


def _check_time(deadline):
    if time.monotonic() > deadline:
        raise RetrievalError("The Chroma capture exceeded its time limit.")


def _check_chunk_id(chunk_id):
    if not isinstance(chunk_id, str) or not chunk_id:
        raise RetrievalError("Chroma returned an invalid chunk ID.")


def _evidence_bytes(chunk_id, document, metadata):
    document_bytes = len(document.encode("utf-8"))
    if document_bytes > MAX_CHUNK_BYTES:
        raise RetrievalError("A Chroma chunk exceeds the lexical text limit.")
    return document_bytes + len(chunk_id.encode("utf-8")) + len(
        json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    )


def _vector_hits(collection, embedding, window):
    result = collection.query(
        query_embeddings=[embedding], n_results=window, include=["distances"],
    )
    hits = {}
    for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True):
        _check_chunk_id(chunk_id)
        if chunk_id in hits:
            raise RetrievalError("Chroma returned a duplicate vector chunk ID.")
        hits[chunk_id] = None if distance_key(distance)[0] else distance
    return hits


def _filtered_vector_hits(collection, embedding, ids, deadline):
    """Rank the eligible IDs Chroma returns even when its filtered ANN omits them."""
    config = collection.configuration or {}
    config = config.get("hnsw") or config.get("spann") or {}
    space = config.get("space") or (collection.metadata or {}).get("hnsw:space", "l2")
    if space not in {"l2", "cosine", "ip"}:
        raise RetrievalError(f"Unsupported Chroma distance metric: {space}.")
    query = np.asarray(embedding, dtype=np.float64)
    if query.ndim != 1 or not np.isfinite(query).all():
        raise RetrievalError("The Chroma query vector is invalid.")
    hits = {}
    for offset in range(0, len(ids), 32):
        _check_time(deadline)
        batch = ids[offset:offset + 32]
        page = collection.get(ids=batch, include=["embeddings"])
        if len(page["ids"]) != len(batch) or set(page["ids"]) != set(batch):
            raise RetrievalError("Chroma returned incomplete embeddings for filtered evidence.")
        vectors = np.asarray(page["embeddings"], dtype=np.float64)
        if vectors.shape != (len(batch), len(query)) or not np.isfinite(vectors).all():
            raise RetrievalError("The stored Chroma vectors do not match the query dimension.")
        if space == "l2":
            difference = vectors - query
            distances = np.einsum("ij,ij->i", difference, difference)
        elif space == "ip":
            distances = 1.0 - vectors @ query
        else:
            norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
            similarity = np.divide(
                vectors @ query, norms, out=np.zeros(len(batch)), where=norms > 0,
            )
            distances = 1.0 - np.clip(similarity, -1.0, 1.0)
        hits.update(zip(page["ids"], map(float, distances), strict=True))
    return hits


def _ordered_distances(distances):
    return sorted(distances, key=lambda key: (distance_key(distances[key]), key))


def _pages(collection, where, deadline):
    options = {"where": where} if where is not None else {}
    offset = 0
    while True:
        _check_time(deadline)
        page = collection.get(
            limit=PAGE_SIZE, offset=offset, include=["documents", "metadatas"], **options
        )
        ids = page["ids"]
        if len(ids) > PAGE_SIZE:
            raise RetrievalError("Chroma exceeded the requested page limit.")
        yield zip(ids, page["documents"], page["metadatas"], strict=True)
        if len(ids) < PAGE_SIZE:
            return
        offset += len(ids)


def _candidate_ranks(collection, query, embedding, window, lexical_weight, where, filter_text, deadline):
    match = _fts_query(query) if lexical_weight > 0 else ""
    filtered = bool(filter_text) or where is not None
    vectors = {}
    if embedding is not None and not filtered:
        vectors = _vector_hits(collection, embedding, window)
    with _lexical_index(bool(match)) as index:
        if (filtered and embedding is not None) or index is not None:
            for page in _pages(collection, where, deadline):
                eligible_ids = []
                for chunk_id, document, metadata in page:
                    _check_chunk_id(chunk_id)
                    document = document or ""
                    if filter_text and not matches_filter(metadata, document, filter_text):
                        continue
                    if index is not None:
                        _evidence_bytes(chunk_id, document, metadata or {})
                        index.execute("INSERT INTO seen VALUES (?)", (chunk_id,))
                        index.execute("INSERT INTO evidence VALUES (?, ?)", (chunk_id, document))
                    eligible_ids.append(chunk_id)
                if filtered and embedding is not None and eligible_ids:
                    if len(set(eligible_ids)) != len(eligible_ids):
                        raise RetrievalError("Chroma returned a duplicate eligible chunk ID.")
                    vectors.update(_filtered_vector_hits(collection, embedding, eligible_ids, deadline))
                    vectors = {key: vectors[key] for key in _ordered_distances(vectors)[:window]}
        _check_time(deadline)
        lexical = {}
        if index is not None:
            lexical = {
                key: -score for key, score in index.execute(
                    "SELECT chunk_id, bm25(evidence) AS score FROM evidence "
                    "WHERE evidence MATCH ? ORDER BY score, chunk_id LIMIT ?", (match, window)
                )
            }
        _check_time(deadline)
    return vectors, lexical


def _hydrate(collection, vectors, lexical, include_embeddings, deadline):
    requested = list(dict.fromkeys([*vectors, *lexical]))
    collection_id = str(collection.id)
    evidence = {}
    remaining = MAX_EVIDENCE_BYTES
    for offset in range(0, len(requested), 32):
        _check_time(deadline)
        keys = requested[offset:offset + 32]
        for vector_group in (True, False) if include_embeddings else (False,):
            if include_embeddings:
                group = [key for key in keys if (key in vectors) == vector_group]
            else:
                group = keys
            if not group:
                continue
            include = ["documents", "metadatas"] + (["embeddings"] if vector_group else [])
            values = collection.get(ids=group, include=include)
            embeddings = values["embeddings"] if vector_group else [None] * len(values["ids"])
            for key, doc, meta, vector in zip(
                values["ids"], values["documents"], values["metadatas"], embeddings, strict=True
            ):
                if key in evidence or key not in group:
                    raise RetrievalError("Chroma returned unexpected evidence IDs.")
                doc, meta = doc or "", meta or {}
                remaining -= _evidence_bytes(key, doc, meta)
                if vector is not None:
                    remaining -= 16 * len(vector)
                if remaining < 0:
                    raise RetrievalError("The Chroma candidates exceed the evidence size limit.")
                evidence[key] = {
                    "collection_id": collection_id, "chunk_id": key, "document": doc,
                    "meta": meta, "distance": vectors.get(key),
                }
                if vector is not None:
                    evidence[key]["embedding"] = list(map(float, vector))
                if key in lexical:
                    evidence[key]["bm25_score"] = lexical[key]
    if len(evidence) != len(requested):
        raise RetrievalError("Chroma returned incomplete evidence for the captured candidate IDs.")
    _check_time(deadline)
    return evidence


def search_keywords(rewritten, synonyms):
    words = set(re.findall(r"[^\W_]+", rewritten.casefold()))
    words.difference_update("is a an and the of in on at by with for to from".split())
    for key, variants in synonyms.items():
        words.add(key.casefold())
        for values in (variants.values() if isinstance(variants, dict) else [variants]):
            if isinstance(values, list):
                words.update(value.casefold() for value in values if isinstance(value, str) and value)
    return sorted(words)


def _boost(ranked, words, phrase, keyword_weight, phrase_weight):
    if not ranked or (keyword_weight <= 0 and phrase_weight <= 0):
        return
    keyword_patterns = [re.compile(rf"\b{re.escape(word)}\b") for word in words]
    counts = []
    for row in ranked:
        doc = row["document"].casefold()
        normalized = re.sub(r"(?<=\w)-\s*(?=\w)", "", doc)
        keywords = sum(sum(1 for _ in pattern.finditer(normalized)) for pattern in keyword_patterns)
        counts.append((keywords, doc.count(phrase) if phrase else 0))
    max_keywords = max(count[0] for count in counts) or 1
    max_phrase = max(count[1] for count in counts) or 1
    for row, (keyword_count, phrase_count) in zip(ranked, counts):
        magnitude = (keyword_weight * keyword_count / max_keywords
                     + phrase_weight * phrase_count / max_phrase)
        row["final_score"] += magnitude / 61
    ranked.sort(key=lambda row: (-row["final_score"], row["collection_id"], row["chunk_id"]))


def _source_group(row):
    meta = row["meta"]
    path = meta.get("file_path") or meta.get("file_name")
    if isinstance(path, str) and (meta.get("file_path") or "/" in path or "\\" in path):
        return row["collection_id"], "path", os.path.normcase(path)
    return row["collection_id"], "chunk", row["chunk_id"]


def _select(ranked, top_k, additional, diversity, deadline):
    selected = ranked[:top_k]
    if not additional or not ranked[top_k:]:
        return selected
    seen = {_source_group(row) for row in selected}
    remaining = [(row, _source_group(row)) for row in ranked[top_k:]]
    vectors = {
        row["chunk_id"]: np.asarray(row["embedding"])
        for row in ranked if row.get("embedding") is not None
    }
    highest_similarity = {}

    def compare_to(chosen):
        vector = vectors.get(chosen["chunk_id"])
        if vector is None:
            return
        for row, group in remaining:
            chunk = row["chunk_id"]
            if chunk in vectors and group not in seen:
                similarity = float(np.dot(vector, vectors[chunk]))
                highest_similarity[chunk] = max(
                    highest_similarity.get(chunk, -math.inf), similarity,
                )

    def score(row):
        chunk = row["chunk_id"]
        if chunk not in vectors:
            return row["final_score"]
        return diversity * row["final_score"] - (1 - diversity) * highest_similarity.get(chunk, 0)

    for row in selected:
        _check_time(deadline)
        compare_to(row)
    for _ in range(additional):
        _check_time(deadline)
        eligible = ((row, group) for row, group in remaining if group not in seen)
        best = max(eligible, key=lambda item: score(item[0]), default=None)
        if best is None:
            break
        row, group = best
        seen.add(group)
        selected.append(row)
        compare_to(row)
    return selected


def capture(
    collection, query, *, embedding=None, vector_weight=0, lexical_weight=1, where=None,
    filter_text="", top_k=5, additional=0, diversity=0.5, keyword_weight=0,
    phrase_weight=0, words=(), phrase="", exact_window=False,
):
    """Capture and rank one collection snapshot under the caller's store lock."""
    if top_k <= 0 or top_k + additional > 1000 or additional < 0:
        raise RetrievalError("The requested Chroma candidate limit is outside 1–1,000.")
    if not 0 <= diversity <= 1:
        raise RetrievalError("Chroma search diversity must be between zero and one.")
    if any(not math.isfinite(weight) or weight < 0 for weight in (
        vector_weight, lexical_weight, keyword_weight, phrase_weight
    )):
        raise RetrievalError("Chroma search weights must be finite and non-negative.")
    if vector_weight > 0 and embedding is None:
        raise RetrievalError("The embedding provider returned no query vector.")
    deadline = time.monotonic() + CAPTURE_SECONDS
    window = top_k if exact_window else max(50, top_k + additional)
    vectors, lexical = _candidate_ranks(
        collection, query, embedding, window, lexical_weight, where,
        filter_text.strip().casefold(), deadline,
    )
    evidence = _hydrate(collection, vectors, lexical, bool(additional), deadline)
    ranked = fuse_candidates(
        (evidence[key] for key in _ordered_distances(vectors)),
        (evidence[key] for key in lexical),
        vector_weight=vector_weight, lexical_weight=lexical_weight,
    )
    _boost(ranked, words, phrase.casefold(), keyword_weight, phrase_weight)
    selected = _select(ranked, top_k, additional, diversity, deadline)
    _check_time(deadline)
    return selected


def _missing_ids(collection, keys):
    present = set()
    for start in range(0, len(keys), PAGE_SIZE):
        present.update(collection.get(ids=keys[start:start + PAGE_SIZE], include=[])["ids"])
    return [key for key in keys if key not in present]


def _watcher_write(selection, embedding_model, ids, values=None):
    with store_lock(selection["path"]):
        collection = _open_selection(selection)
        _check_selection(collection, selection["expected_id"], None)
        validate_embedding_compatibility(collection, embedding_model, for_write=True)
        missing = _missing_ids(collection, ids)
        if values is not None and missing:
            missing_set = set(missing)
            keep = [index for index, key in enumerate(ids) if key in missing_set]
            collection.add(ids=missing, **{
                name: [items[index] for index in keep] for name, items in values.items()
            })
        return missing


def watcher_chunks(selection, embedding_model, ids, values=None):
    """Check missing IDs or add the still-missing chunks through a fresh writer."""
    return _run_operation(
        "watcher_write", selection, embedding_model=embedding_model, ids=ids, values=values,
    )


def _main():
    request = json.load(sys.stdin)
    operation = request.pop("operation")
    selection = request.pop("selection")
    operations = {"inspect": _inspect, "capture": _capture_selection, "watcher_write": _watcher_write}
    try:
        response = {"result": operations[operation](selection, **request)}
    except Exception as error:
        response = {"error": str(error), "contract": isinstance(error, EmbeddingContractError)}
    print(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    _main()
