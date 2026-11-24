import ast
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _install_dependency_stubs(monkeypatch):
    """Keep legacy module imports inert while exercising their public seams."""

    ollama = types.ModuleType("ollama")
    ollama.chat = lambda **kwargs: iter(())
    ollama.embeddings = lambda **kwargs: {"embedding": [0.0]}
    ollama.Client = lambda **kwargs: SimpleNamespace()

    groq = types.ModuleType("groq")
    groq.Groq = lambda **kwargs: SimpleNamespace()

    chromadb = types.ModuleType("chromadb")
    chromadb.PersistentClient = lambda **kwargs: SimpleNamespace(
        get_collection=lambda name, **options: SimpleNamespace()
    )
    chromadb.config = types.ModuleType("chromadb.config")
    chromadb.config.DEFAULT_TENANT = "default_tenant"
    chromadb.config.DEFAULT_DATABASE = "default_database"
    chromadb.config.Settings = lambda **kwargs: SimpleNamespace(**kwargs)

    pydub = types.ModuleType("pydub")
    pydub.AudioSegment = SimpleNamespace(from_file=lambda *args, **kwargs: None)
    playback = types.ModuleType("pydub.playback")
    playback.play = lambda audio: None

    speech = types.ModuleType("speech_recognition")
    speech.Recognizer = object
    speech.Microphone = object
    speech.UnknownValueError = type("UnknownValueError", (Exception,), {})
    speech.RequestError = type("RequestError", (Exception,), {})

    gtts = types.ModuleType("gtts")
    gtts.gTTS = lambda **kwargs: SimpleNamespace(write_to_fp=lambda fp: None)

    nest_asyncio = types.ModuleType("nest_asyncio")
    nest_asyncio.apply = lambda: None
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None

    numpy = types.ModuleType("numpy")
    numpy.dot = lambda left, right: 0.0
    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = lambda corpus: SimpleNamespace(
        get_scores=lambda query: [0.0 for _ in corpus]
    )

    modules = {
        "ollama": ollama,
        "groq": groq,
        "chromadb": chromadb,
        "chromadb.config": chromadb.config,
        "pydub": pydub,
        "pydub.playback": playback,
        "speech_recognition": speech,
        "gtts": gtts,
        "nest_asyncio": nest_asyncio,
        "dotenv": dotenv,
        "numpy": numpy,
        "rank_bm25": rank_bm25,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_module(monkeypatch, filename, module_name):
    """Execute only inert imports/definitions, never import a maintained backend.

    AGENTS.md forbids importing the legacy maintained entry points in a test
    process because their historical module scope opened Chroma and started
    workers.  The production files now defer those effects, but keep this
    test boundary explicit: select imports, assignments and definitions from
    the AST and omit module-scope expressions and ``__main__`` execution.
    """

    _install_dependency_stubs(monkeypatch)
    path = ROOT / filename
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    allowed = tuple(
        node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        )
    )
    selected = ast.fix_missing_locations(ast.Module(body=list(allowed), type_ignores=[]))
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module_name, module)
    exec(compile(selected, str(path), "exec"), module.__dict__)
    return module


def _router_fixture(provider, calls):
    class Router:
        def catalogue(self, refresh=False):
            calls.append((provider, "catalogue", refresh))
            return [
                SimpleNamespace(
                    id="phi3:latest" if provider == "ollama" else "groq-model",
                    capabilities={"stream", "json_object"},
                )
            ]

        def generate(self, **request):
            calls.append((provider, "generate", request))
            return SimpleNamespace(
                text="Synthetic answer.",
                preferred_model=request.get("preferred"),
                actual_model=request.get("preferred"),
                reason="preferred model",
            )

    return Router()


def test_explicit_ollama_mode_does_not_construct_groq_and_lists_models(monkeypatch):
    backend = _load_module(
        monkeypatch, "groq_lama_chromadb_RAG_ETTS.py", "legacy_cli_under_test"
    )
    calls = []
    output = io.StringIO()

    def factory(provider):
        calls.append((provider, "factory"))
        return _router_fixture(provider, calls)

    assert (
        backend.main(
            ["--provider", "ollama", "--list-models"],
            router_factory=factory,
            stdout=output,
        )
        == 0
    )

    assert [entry for entry in calls if entry[1] == "factory"] == [("ollama", "factory")]
    assert "phi3:latest" in output.getvalue()


def test_default_mode_uses_groq_without_ollama_generation_client(monkeypatch):
    backend = _load_module(
        monkeypatch, "groq_lama_chromadb_RAG_ETTS.py", "legacy_cli_default_under_test"
    )
    calls = []
    output = io.StringIO()
    monkeypatch.setattr(
        backend,
        "get_relevant_context_hybrid",
        lambda query, **options: "Evidence from local document.",
    )

    def factory(provider):
        calls.append((provider, "factory"))
        return _router_fixture(provider, calls)

    # The speech-suppression marker keeps this network-free CLI test from
    # starting the downstream audio worker; production answers still queue
    # speech only after the router accepts a result.
    answers = iter(["ssssquestion", "exit"])
    assert (
        backend.main(
            [],
            router_factory=factory,
            input_fn=lambda prompt: next(answers),
            stdout=output,
        )
        == 0
    )

    assert [entry for entry in calls if entry[1] == "factory"] == [("groq", "factory")]
    assert "Synthetic answer." in output.getvalue()
    assert "actual_model" in output.getvalue()
    generation_calls = [entry[2] for entry in calls if entry[1] == "generate"]
    assert "Evidence from local document." in generation_calls[0]["messages"][-1]["content"]


def test_offline_search_parses_before_provider_or_collection_creation(monkeypatch):
    backend = _load_module(
        monkeypatch, "groq_lama_chromadb_RAG_ETTS.py", "legacy_cli_offline_under_test"
    )
    calls = []
    output = io.StringIO()

    def factory(provider):
        calls.append((provider, "factory"))
        pytest.fail("offline search must not construct a generation router")

    def open_collection(path, name):
        calls.append(("collection", path, name))
        return object()

    def search_documents(collection, query, limit):
        calls.append(("search", query, limit))
        return [{"id": "doc-1", "document": "Offline evidence", "metadata": {}}]

    assert (
        backend.main(
            ["--offline-search", "needle"],
            router_factory=factory,
            collection_opener=open_collection,
            searcher=search_documents,
            stdout=output,
        )
        == 0
    )
    assert calls[0][0] == "collection"
    assert "Offline evidence" in output.getvalue()


def test_legacy_rewrite_uses_shared_router_and_keeps_original_query_on_404(monkeypatch):
    backend = _load_module(
        monkeypatch, "context_retrieval.py", "legacy_context_under_test"
    )

    class FakeRouter:
        def generate(self, **request):
            assert request["preferred"] == backend.groq_model
            error = RuntimeError("model_not_found")
            error.code = "model_unavailable"
            raise error

    generation_models = types.ModuleType("generation_models")
    generation_models.groq_router = lambda **kwargs: FakeRouter()
    monkeypatch.setitem(sys.modules, "generation_models", generation_models)

    original = "raw legacy query"
    rewritten, synonyms = backend.rewrite_input_and_generate_synonyms(original)

    assert rewritten == original
    assert synonyms == {}


def test_legacy_rewrite_caches_router_and_requests_domain_synonyms(monkeypatch):
    backend = _load_module(
        monkeypatch, "context_retrieval.py", "legacy_context_prompt_under_test"
    )
    calls = {"factory": 0, "requests": []}

    class FakeRouter:
        def generate(self, **request):
            calls["requests"].append(request)
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "rephrased": "rewritten medical query",
                        "keywords": {"medical": {"synonyms": ["clinical"]}},
                    }
                )
            )

    def factory():
        calls["factory"] += 1
        return FakeRouter()

    generation_models = types.ModuleType("generation_models")
    generation_models.groq_router = factory
    monkeypatch.setitem(sys.modules, "generation_models", generation_models)

    assert backend.rewrite_input_and_generate_synonyms("first query")[0]
    assert backend.rewrite_input_and_generate_synonyms("second query")[0]
    assert calls["factory"] == 1
    prompt = calls["requests"][0]["messages"][0]["content"].lower()
    for phrase in ("medical context", "synonyms", "spelling variants", "plural"):
        assert phrase in prompt
