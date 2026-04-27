import json
import os
import subprocess
import sys
import webbrowser

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QListWidgetItem, QMessageBox

# Static script — no user data interpolated.  All params arrive via the
# _DS_ARGS env var (JSON), so there is no code-injection surface.
_SEARCH_SCRIPT = """
import json, os, chromadb, ollama, re, numpy as np, sys
from groq import Groq
from rank_bm25 import BM25Okapi
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

_a = json.loads(os.environ["_DS_ARGS"])
path            = _a["path"]
col_name        = _a["col_name"]
query           = _a["query"]
embedding_model = _a["embedding_model"]
rewrite_model   = _a["rewrite_model"]
top_k           = int(_a["top_k"])
alpha           = float(_a["alpha"])
beta            = float(_a["beta"])
gamma           = float(_a["gamma"])
delta           = float(_a["delta"])
mode            = _a["mode"]
include         = _a["include"]

def rewrite_input_and_generate_synonyms(user_input):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        system_prompt = (
            "You are a helpful assistant. Your tasks are:\\n"
            "1) Rephrase the given input to make it clearer in one sentence.\\n"
            "2) Provide synonyms, spelling variants, plural/singular forms for keywords.\\n"
            'Respond in JSON format:\\n{"rephrased": "[sentence]", "keywords": {"[word]": {"synonyms": [], "spelling_variants": [], "plural_singular": [], "parts_of_speech": [], "related_terms": []}}}'
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Rewrite and generate synonyms for: "{user_input}"'},
            ],
            model=rewrite_model,
            temperature=0.7,
            stream=False,
            response_format={"type": "json_object"},
        )
        response_json = chat_completion.choices[0].message.content.strip()
        data = json.loads(response_json)
        return data.get("rephrased", user_input), data.get("keywords", {})
    except Exception:
        return user_input, {}

def build_keywords(rewritten, synonym_dict):
    non_keywords = {"is","a","an","and","the","of","in","on","at","by","with","for","to","from"}
    kws = [w for w in rewritten.lower().split() if w not in non_keywords]
    for key, details in synonym_dict.items():
        kws.append(key)
        for field in ["synonyms","spelling_variants","plural_singular","parts_of_speech","related_terms"]:
            vals = details.get(field) or []
            kws.extend(vals)
    return list(set(kws))

def normalize_scores(items, key):
    max_v = max([x.get(key,0) for x in items], default=1) or 1
    for x in items:
        x[key] = x.get(key,0) / max_v

def phrase_boost(document, rewritten):
    if not rewritten:
        return 0.0
    phrase = rewritten.lower()
    doc = (document or "").lower()
    return doc.count(phrase)

client = chromadb.PersistentClient(path=path, settings=Settings(), tenant=DEFAULT_TENANT, database=DEFAULT_DATABASE)
col = client.get_collection(col_name)

if mode == "phrase":
    emb = ollama.embeddings(model=embedding_model, prompt=query, keep_alive=-1)["embedding"]
    res = col.query(query_embeddings=[emb], n_results=top_k, include=include)
    formatted = []
    docs = res.get("documents",[[]])[0]
    metas = res.get("metadatas",[[]])[0]
    dists = res.get("distances",[[]])[0]
    for meta, doc, dist in zip(metas, docs, dists):
        similarity = 1.0 - dist if isinstance(dist,(int,float)) else None
        formatted.append({
            "file_name": meta.get("file_name","Unknown") if isinstance(meta, dict) else "Unknown",
            "file_path": meta.get("file_path") if isinstance(meta, dict) else None,
            "document": doc,
            "distance": dist,
            "similarity": similarity,
            "metadata": meta,
            "final_score": similarity if similarity is not None else None,
        })
    print(json.dumps({"results": formatted}))
    sys.exit(0)

rewritten_input, synonym_dict = rewrite_input_and_generate_synonyms(query)
emb = ollama.embeddings(model=embedding_model, prompt=rewritten_input, keep_alive=-1)["embedding"]
res = col.query(query_embeddings=[emb], n_results=50, include=["documents","metadatas","distances"])

vector_results = [
    {"meta": meta, "document": doc, "vector_score": 1.0 - dist, "distance": dist}
    for meta, doc, dist in zip(res.get("metadatas",[[]])[0], res.get("documents",[[]])[0], res.get("distances",[[]])[0])
]

keywords = build_keywords(rewritten_input, synonym_dict)
keyword_results = []
phrase_results = []
for meta, doc in zip(res.get("metadatas",[[]])[0], res.get("documents",[[]])[0]):
    normalized_doc = re.sub(r"(?<=\\w)-\\s*(?=\\w)", "", (doc or "").lower())
    match_score = sum(len(re.findall(rf"\\b{re.escape(kw)}\\b", normalized_doc)) + len(re.findall(rf"\\b{re.escape(kw)}\\b", str(meta.get("file_name","")).lower())) for kw in keywords)
    if match_score > 0:
        keyword_results.append({"meta": meta, "document": doc, "keyword_score": match_score})
    phrase_score = phrase_boost(doc, rewritten_input)
    if phrase_score > 0:
        phrase_results.append({"meta": meta, "document": doc, "phrase_score": phrase_score})

bm25_corpus = [doc or "" for doc in res.get("documents",[[]])[0]]
bm25 = BM25Okapi([d.split() for d in bm25_corpus]) if bm25_corpus else None
bm25_scores = bm25.get_scores(rewritten_input.split()) if bm25 else []
bm25_results = [
    {"meta": meta, "document": doc, "bm25_score": score}
    for meta, doc, score in zip(res.get("metadatas",[[]])[0], res.get("documents",[[]])[0], bm25_scores)
]

normalize_scores(vector_results, "vector_score")
normalize_scores(keyword_results, "keyword_score")
normalize_scores(bm25_results, "bm25_score")
normalize_scores(phrase_results, "phrase_score")

combined = {}
for item in vector_results:
    name = item.get("meta",{}).get("file_name","Unknown")
    combined[name] = {"meta": item.get("meta",{}), "document": item.get("document",""), "final_score": alpha*item.get("vector_score",0), "distance": item.get("distance")}

for item in keyword_results:
    name = item.get("meta",{}).get("file_name","Unknown")
    combined.setdefault(name, {"meta": item.get("meta",{}), "document": item.get("document",""), "final_score": 0})["final_score"] += beta*item.get("keyword_score",0)

for item in bm25_results:
    name = item.get("meta",{}).get("file_name","Unknown")
    combined.setdefault(name, {"meta": item.get("meta",{}), "document": item.get("document",""), "final_score": 0})["final_score"] += gamma*item.get("bm25_score",0)

for item in phrase_results:
    name = item.get("meta",{}).get("file_name","Unknown")
    combined.setdefault(name, {"meta": item.get("meta",{}), "document": item.get("document",""), "final_score": 0})["final_score"] += delta*item.get("phrase_score",0)

sorted_results = sorted(combined.values(), key=lambda x: x.get("final_score",0), reverse=True)[:top_k]
formatted = []
for item in sorted_results:
    meta = item.get("meta") or {}
    doc = item.get("document") or ""
    dist = item.get("distance")
    similarity = None
    if dist is not None and isinstance(dist,(int,float)):
        similarity = 1.0 - dist
    formatted.append({
        "file_name": meta.get("file_name","Unknown") if isinstance(meta, dict) else "Unknown",
        "file_path": meta.get("file_path") if isinstance(meta, dict) else None,
        "document": doc,
        "distance": dist,
        "similarity": similarity,
        "metadata": meta,
        "final_score": item.get("final_score"),
    })

print(json.dumps({"results": formatted}))
"""


class DirectSearchSubprocessWorker(QThread):
    results_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        query,
        chroma_path,
        collection_name,
        embed_model,
        rewrite_model,
        n_results,
        include,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        delta=0.2,
        mode="hybrid",
    ):
        super().__init__()
        self.query = query
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.rewrite_model = rewrite_model
        self.n_results = n_results
        self.include = include
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.mode = (mode or "hybrid").lower()

    def run(self):
        alpha = float(self.alpha)
        beta = float(self.beta)
        gamma = float(self.gamma)
        delta = float(self.delta)
        mode = self.mode
        ds_args = json.dumps({
            "path": self.chroma_path,
            "col_name": self.collection_name,
            "query": self.query,
            "embedding_model": self.embed_model,
            "rewrite_model": self.rewrite_model,
            "top_k": int(self.n_results),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "mode": mode,
            "include": self.include,
        })
        try:
            env = {**os.environ, "_DS_ARGS": ds_args}
            result = subprocess.run(
                [sys.executable, "-c", _SEARCH_SCRIPT],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Subprocess query failed rc={result.returncode} stderr={result.stderr.strip()} stdout={result.stdout.strip()}"
                )
            payload = json.loads(result.stdout)
            formatted = payload.get("results") or []
            if not formatted:
                raise RuntimeError("No results returned from hybrid search")
            self.results_ready.emit(formatted)
        except Exception as e:
            import traceback as _tb

            _tb.print_exc()
            self.error_occurred.emit(str(e))


class DirectSearchMixin:
    """Direct ChromaDB search helpers split from main GUI."""

    def direct_chromadb_search(self):
        print("[DirectSearchMixin.direct_chromadb_search] start", flush=True)
        query = self.direct_search_input.text().strip()
        if not query or not self.collection:
            print(
                "[DirectSearchMixin.direct_chromadb_search] abort: query_present=%s collection_present=%s"
                % (bool(query), bool(self.collection)),
                flush=True,
            )
            return

        print(
            "[DirectSearchMixin.direct_chromadb_search] query_len=%s collection=%s"
            % (len(query), getattr(self.collection, "name", "<no-name>")),
            flush=True,
        )
        # Self-test path: one-shot subprocess call to see if Chroma returns or crashes
        if getattr(self, "self_test_enabled", False):
            if getattr(self, "_self_test_sync_active", False):
                print("[SelfTest] Sync search already running; skipping re-entry", flush=True)
                return
            self._self_test_sync_active = True
            # Ensure we only run once
            self.self_test_enabled = False
            print("[SelfTest] Running subprocess direct search (one-shot)", flush=True)
            try:
                res = self._query_via_subprocess(query, n_results=1, include=["distances"])
                print(
                    "[SelfTest-subprocess] result keys=%s lens=%s"
                    % (
                        list(res.keys()),
                        {k: (len(v[0]) if isinstance(v, list) and v else "n/a") for k, v in res.items()},
                    ),
                    flush=True,
                )
            except Exception as e:
                import traceback as _tb

                print("[SelfTest] sync query error: %s" % e, flush=True)
                _tb.print_exc()
            finally:
                self._self_test_sync_active = False
            return

        # Normal path: run search via subprocess to avoid native crash in worker
        # Sync top-k from direct control before saving settings
        if hasattr(self, "direct_topk_spin"):
            self.settings["top_k"] = int(self.direct_topk_spin.value())

        self.update_settings_from_ui()
        top_k = int(self.settings.get("top_k", 5))
        # Normalize hybrid weights so they sum to 1
        alpha_raw = float(self.settings.get("hybrid_alpha", 0.5))
        beta_raw = float(self.settings.get("hybrid_beta", 0.3))
        gamma_raw = float(self.settings.get("hybrid_gamma", 0.2))
        delta_raw = float(self.settings.get("hybrid_delta", 0.2))
        total = alpha_raw + beta_raw + gamma_raw + delta_raw
        if total <= 0:
            alpha = beta = gamma = 0.0
            delta = 1.0  # fallback to phrase if all zero/negative
            if hasattr(self, "statusBar"):
                self.statusBar.showMessage("Hybrid weights invalid; using delta=1.0 fallback", 4000)
        else:
            alpha = alpha_raw / total
            beta = beta_raw / total
            gamma = gamma_raw / total
            delta = delta_raw / total
            if abs(total - 1.0) > 1e-6 and hasattr(self, "statusBar"):
                self.statusBar.showMessage(f"Normalized hybrid weights (sum was {total:.2f})", 3000)
        print(
            "[DirectSearchMixin.direct_chromadb_search] settings embedding_model=%s top_k=%s weights=(%.2f, %.2f, %.2f, %.2f)"
            % (self.settings.get("embedding_model"), top_k, alpha, beta, gamma, delta),
            flush=True,
        )
        self.statusBar.showMessage("Searching...")
        # Run subprocess query off the UI thread
        self._direct_worker = DirectSearchSubprocessWorker(
            query=query,
            chroma_path=self.settings.get("chromadb_path"),
            collection_name=getattr(self.collection, "name", self.settings.get("collection_name")),
            embed_model=self.settings.get("embedding_model"),
            rewrite_model=self.settings.get("groq_rewrite_model", self.settings.get("model_name", "")),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            mode=self.settings.get("direct_search_mode", "hybrid"),
        )
        self._direct_worker.results_ready.connect(self._on_direct_worker_results)
        self._direct_worker.error_occurred.connect(self.on_error)
        self._direct_worker.finished.connect(lambda: self.statusBar.showMessage("Search complete", 3000))
        self._direct_worker.start()

    def apply_direct_search_view(self):
        results = getattr(self, "_direct_results", []) or []
        filt = getattr(self, "direct_filter_input", None)
        sort_combo = getattr(self, "direct_sort_combo", None)
        filter_text = filt.text().strip().lower() if filt else ""
        view = []
        for r in results:
            hay = (r.get("file_name", "") + " " + (r.get("document", "") or "")).lower()
            if filter_text and filter_text not in hay:
                continue
            view.append(r)
        if sort_combo:
            mode = sort_combo.currentText()
            if mode == "Distance ↑":
                view.sort(key=lambda x: x.get("distance", 0))
            elif mode == "Distance ↓":
                view.sort(key=lambda x: x.get("distance", 0), reverse=True)
            elif mode == "File A→Z":
                view.sort(key=lambda x: x.get("file_name", ""))
        print("[DirectSearchMixin.apply_direct_search_view] rendering %s items" % len(view), flush=True)
        self.on_search_results(view)

    def on_search_results(self, results):
        print(
            "[DirectSearchMixin.on_search_results] received %s results type=%s" % (len(results), type(results)),
            flush=True,
        )
        self.search_results_list.clear()

        for i, res in enumerate(results):
            file_name = res.get("file_name", "Unknown")
            sim = res.get("similarity")
            dist = res.get("distance")
            sim_display = f"{sim:.4f}" if isinstance(sim, (int, float)) else "n/a"
            dist_display = f"{dist:.4f}" if isinstance(dist, (int, float)) else "n/a"
            doc = res.get("document", "")
            preview = (doc or "")[:180].replace("\n", " ")
            line = f"{file_name}\n  sim={sim_display} dist={dist_display}\n  {preview}"
            item = QListWidgetItem(line)
            item.setData(Qt.UserRole, res)
            self.search_results_list.addItem(item)

        self.statusBar.showMessage(f"Found {len(results)} results", 3000)
        print("[DirectSearchMixin.on_search_results] table updated", flush=True)

    def _on_direct_worker_results(self, results):
        # Store and render with sort/filter
        self._direct_results = results
        self.apply_direct_search_view()

    def update_search_result_detail(self, item):
        if not item:
            return
        data = item.data(Qt.UserRole) or {}
        file_name = data.get("file_name", "Unknown")
        file_path = data.get("file_path") or file_name
        dist = data.get("distance")
        sim = data.get("similarity")
        doc = data.get("document", "") or ""
        meta = data.get("metadata") or {}
        detail = (
            f"File: {file_name}\n"
            f"Path: {file_path}\n"
            f"Similarity: {sim}\n"
            f"Distance: {dist}\n\n"
            f"Snippet:\n{doc[:1200]}\n\n"
            f"Metadata:\n{json.dumps(meta, indent=2, default=str)}"
        )
        if hasattr(self, "search_result_detail"):
            self.search_result_detail.setPlainText(detail)

    _ALLOWED_OPEN_EXTENSIONS = {
        ".pdf", ".txt", ".md", ".docx", ".doc", ".rtf",
        ".csv", ".json", ".xml", ".html", ".htm",
        ".py", ".js", ".ts", ".java", ".cs", ".cpp", ".c", ".h",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg",
        ".mp3", ".mp4", ".wav", ".ogg",
    }

    def open_search_result_file(self, item):
        if not item:
            return
        from pathlib import Path
        data = item.data(Qt.UserRole) or {}
        raw = data.get("file_path") or data.get("file_name")
        print(f"[DirectSearch] open_search_result_file raw={raw!r}", flush=True)
        if not raw:
            QMessageBox.information(self, "Open File", "No file path available.")
            return
        try:
            path = Path(raw).resolve()
        except Exception:
            QMessageBox.warning(self, "Open File", "Invalid file path.")
            return
        if not path.is_file():
            QMessageBox.warning(self, "Open File", f"File not found:\n{path}")
            return
        if path.suffix.lower() not in self._ALLOWED_OPEN_EXTENSIONS:
            QMessageBox.warning(self, "Open File", f"File type not allowed to open:\n{path.suffix}")
            return
        # Use explorer only — avoids cmd/start and webbrowser which can
        # execute UNC paths, .lnk, .exe, and file:// script URLs.
        try:
            subprocess.Popen(["explorer", str(path)])
        except Exception as e:
            QMessageBox.warning(self, "Open File", f"Failed to open file:\n{e}")

    def _on_search_worker_finished(self):
        print("[DirectSearchMixin._on_search_worker_finished] search worker finished", flush=True)

    def _on_search_worker_destroyed(self, obj):
        print("[DirectSearchMixin._on_search_worker_destroyed] worker destroyed: %s" % obj, flush=True)

    def show_search_result_detail(self, item):
        print("[DirectSearchMixin.show_search_result_detail] invoked", flush=True)
        self.update_search_result_detail(item)

    # ------------------------------------------------------------------
    # Subprocess helper
    # ------------------------------------------------------------------
    def _query_via_subprocess(self, query, n_results=5, include=None):
        """Run a minimal Chroma query in a subprocess and return parsed results."""
        include = include or ["documents", "metadatas", "distances"]
        chroma_path = self.settings.get("chromadb_path")
        col_name = getattr(self.collection, "name", self.settings.get("collection_name"))
        embed_model = self.settings.get("embedding_model")
        include_literal = "[" + ",".join(repr(x) for x in include) + "]"
        script = f"""
import json, chromadb, ollama
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
path = {repr(chroma_path)}
col_name = {repr(col_name)}
query = {repr(query)}
emb = ollama.embeddings(model={repr(embed_model)}, prompt=query, keep_alive=-1)["embedding"]
client = chromadb.PersistentClient(path=path, settings=Settings(), tenant=DEFAULT_TENANT, database=DEFAULT_DATABASE)
col = client.get_collection(col_name)
res = col.query(query_embeddings=[emb], n_results={int(n_results)}, include={include_literal})
print(json.dumps(res))
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Subprocess query failed rc={result.returncode} stderr={result.stderr.strip()} stdout={result.stdout.strip()}")
            if not result.stdout:
                raise RuntimeError("Subprocess query produced no output")
            try:
                return json.loads(result.stdout)
            except Exception as je:
                raise RuntimeError(f"Failed to parse subprocess JSON: {je}; raw={result.stdout!r}")
        except Exception as e:
            import traceback as _tb

            print("[SelfTest-subprocess] failed to run: %s" % e, flush=True)
            _tb.print_exc()
            raise
