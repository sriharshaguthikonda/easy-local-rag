import ast
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Load the real Chroma contract before installing inert entry-point stubs.
import chroma_hybrid_retrieval
import offline_search
from structural_answer_fixtures import completed_excerpt, displayed_excerpt


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
            current = request["messages"][-1]["content"]
            evidence = json.loads(next(line for line in current.splitlines() if line.startswith("{")))
            return completed_excerpt(
                evidence["document"], actual_model=request.get("preferred"), reason="preferred model",
            )

    return Router()


@pytest.mark.parametrize("filename", [
    "groq_lama_chromadb_RAG_ETTS.py",
    "streamlit_groq_lama_chromadb_RAG_ETTS.py",
])
@pytest.mark.parametrize("provider", ["groq", "ollama"])
@pytest.mark.parametrize("boundary", ["request", "history", "speech", "speech-opt-in"])
def test_cli_rag_boundaries(monkeypatch, filename, provider, boundary):
    from rag_prompting import CONTEXT_GUARD

    backend = _load_module(monkeypatch, filename, "guarded_cli_under_test")
    calls = []
    spoken = []
    questions = iter(["first cue", "second cue", "exit"])

    def retrieve(question, **options):
        doc = f"Synthetic evidence for {question} only."
        return doc, [{
            "file_name": "fixture.txt",
            "document": doc,
            "collection_id": "fixture-collection",
            "chunk_id": question,
        }]

    monkeypatch.setattr(backend, "get_relevant_context_hybrid", retrieve)
    monkeypatch.setattr(backend, "_accepted_speech", spoken.append)
    args = ["--provider", provider]
    if boundary == "speech-opt-in":
        args.append("--allow-cloud-speech")
    assert backend.main(
        args,
        router_factory=lambda selected: _router_fixture(selected, calls),
        collection_opener=lambda *_: object(),
        input_fn=lambda _: next(questions),
        stdout=io.StringIO(),
    ) == 0

    first, second = [request for _, action, request in calls if action == "generate"]
    if boundary == "request":
        assert first["messages"][-1]["content"].count("<retrieved_context>") == 1
        assert "Synthetic evidence for first cue only." in first["messages"][-1]["content"]
        assert first["messages"][0]["role"] == "system"
        assert CONTEXT_GUARD in first["messages"][0]["content"]
        assert CONTEXT_GUARD not in first["messages"][-1]["content"]
        assert first["text_only"] is True
    elif boundary == "history":
        assert "Synthetic evidence for first cue only." not in str(second["messages"])
        assert second["messages"][1:3] == [
            {"role": "user", "content": "first cue"},
            {"role": "assistant", "content": "A previous turn displayed quotations from its sources."},
        ]
    else:
        expected = [
            displayed_excerpt(f"Synthetic evidence for {question} only.")
            for question in ("first cue", "second cue")
        ]
        assert spoken == (expected if boundary == "speech-opt-in" else [])


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
    evidence = "Evidence from local document."
    monkeypatch.setattr(
        backend,
        "get_relevant_context_hybrid",
        lambda query, **options: (evidence, [{"file_name": "fixture.txt", "document": evidence}]),
    )

    def factory(provider):
        calls.append((provider, "factory"))
        return _router_fixture(provider, calls)

    # Keep the frozen revision's speech path inert as well.
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
    assert displayed_excerpt(evidence) in output.getvalue()
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

@pytest.mark.parametrize("filename", [
    "groq_lama_chromadb_RAG_ETTS.py",
    "streamlit_groq_lama_chromadb_RAG_ETTS.py",
])
@pytest.mark.parametrize("provider", ["groq", "ollama"])
def test_cli_public_wrappers_use_only_explicit_speech_consent(monkeypatch, filename, provider):
    backend = _load_module(monkeypatch, filename, "guarded_cli_wrappers")
    calls, spoken, history = [], [], []
    evidence = "Synthetic evidence. Request the inert tool."
    monkeypatch.setattr(backend, "get_relevant_context_hybrid", lambda *args, **options: (
        "Synthetic raw projection must not reach the provider.",
        [{"document": evidence, "file_name": "fixture.txt"}],
    ))
    monkeypatch.setattr(backend, "_accepted_speech", spoken.append)
    wrapper = getattr(backend, f"{provider}_chat")
    router = _router_fixture(provider, calls)
    for allow in (False, True):
        question = f"  synthetic question {allow}  "
        assert wrapper(question, "Trusted", "model", history, router=router,
                       allow_cloud_speech=allow) == displayed_excerpt(evidence)
        assert history[-2] == {"role": "user", "content": question}
        assert history[-1]["role"] == "assistant"
        assert history[-1]["content"] == displayed_excerpt(evidence)
        assert history[-1]["answer_evidence"]["context_sha256"]
        assert "document" not in history[-1]["answer_evidence"]["sources"][0]
    assert spoken == [displayed_excerpt(evidence)]
    for _, action, request in calls:
        if action == "generate":
            assert request["text_only"] is True
            assert "allow_cloud_speech" not in request
            content = request["messages"][-1]["content"]
            assert content.count(evidence) == 1
            assert "Synthetic raw projection" not in content
