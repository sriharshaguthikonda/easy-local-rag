import ast
from pathlib import Path
from types import SimpleNamespace


class _FakeCollection:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _NoopThread:
    def __init__(self, **_kwargs):
        pass

    def start(self):
        pass


def test_hybrid_context_uses_documents_and_greedy_mmr_without_metadata_text():
    source = Path("streamlit_groq_lama_chromadb_RAG_ETTS.py").read_text(
        encoding="utf-8"
    )
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_relevant_context_hybrid"
    )

    metadata = [
        {"file_name": "a.txt", "modification_time": "1"},
        {"file_name": "b.txt", "modification_time": "2"},
        {"file_name": "c.txt", "modification_time": "3"},
        {"file_name": "d.txt", "modification_time": "4"},
    ]
    collection = _FakeCollection(
        {
            "documents": [["needle alpha", "needle beta", "needle gamma", "needle delta"]],
            "metadatas": [metadata],
            "distances": [[0.0, 0.1, 0.2, 0.3]],
            "embeddings": [[[1, 0], [1, 0], [0, 1], [0, 1]]],
        }
    )
    namespace = {
        "rewrite_input_and_generate_synonyms": lambda text: (text, {}),
        "ollama": SimpleNamespace(
            embeddings=lambda **_kwargs: {"embedding": [1, 0]}
        ),
        "model": "test-model",
        "collection": collection,
        "non_keywords": set(),
        "np": SimpleNamespace(dot=lambda left, right: sum(a * b for a, b in zip(left, right))),
        "threading": SimpleNamespace(Thread=_NoopThread),
        "print_relevant_context": lambda _results: None,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "backend", "exec"), namespace)

    result = namespace["get_relevant_context_hybrid"](
        "needle", top_k=1, additional_unique_files=2, keyword_match=True
    )

    assert collection.calls == [
        {
            "query_embeddings": [[1, 0]],
            "n_results": 50,
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
    ]
    assert result != ("Answer this yourself!", [])
    context, returned_metadata = result
    assert context == "needle alpha\n\nneedle gamma\n\nneedle beta"
    assert returned_metadata == [metadata[0], metadata[2], metadata[1]]
