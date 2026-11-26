import json
import os
import subprocess
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QListWidgetItem, QMessageBox, QTableWidgetItem

from chroma_hybrid_retrieval import (
    CAPTURE_PROCESS_TIMEOUT, canonical_store_path, distance_key, persistent_selection,
)
from GUI_context import rewrite_input_and_generate_synonyms


_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

# Static script — no user data interpolated.  All params arrive via the
# _DS_ARGS env var (JSON), so there is no code-injection surface.
_SEARCH_SCRIPT = """
import json
import os
import sys

import ollama

args = json.loads(os.environ["_DS_ARGS"])
sys.path.insert(0, args["module_path"])
from chroma_hybrid_retrieval import search_keywords, search_path_in_worker, self_test_query

path = args["path"]
name = args["col_name"]
query = args["query"]
model = args["embedding_model"]
mode = args["mode"]
top_k = int(args["top_k"])
expected_id = args.get("collection_id")
rewritten = args.get("rewritten_input", query) if mode == "hybrid" else query


def embed():
    return ollama.embeddings(model=model, prompt=rewritten, keep_alive=-1)["embedding"]


if mode == "self_test":
    result = self_test_query(path, name, model, expected_id, embed, top_k, args["include"])
    print(json.dumps(result, default=lambda value: value.tolist()))
else:
    hybrid = mode == "hybrid"
    rows = search_path_in_worker(
        path, name, query, embedding_model=model, embed=embed, expected_id=expected_id,
        top_k=top_k, filter_text=args.get("filter_text", ""),
        vector_weight=float(args["alpha"]) if hybrid else 1.0,
        lexical_weight=float(args["gamma"]) if hybrid else 0.0,
        keyword_weight=float(args["beta"]) if hybrid else 0.0,
        phrase_weight=float(args["delta"]) if hybrid else 0.0,
        words=search_keywords(rewritten, args.get("synonym_dict", {})),
        phrase=rewritten, exact_window=not hybrid,
    )
    for row in rows:
        meta, distance = row["meta"], row["distance"]
        similarity = None if distance is None else 1.0 - distance
        row.update(
            metadata=meta, file_name=str(meta.get("file_name") or "Unknown"),
            file_path=meta.get("file_path"), similarity=similarity,
        )
        if not hybrid:
            row["final_score"] = similarity
    print(json.dumps({"results": rows}))
"""


class DirectSearchSubprocessWorker(QThread):
    results_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)

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
        rewrite_settings=None,
        filter_text="",
        collection_id=None,
    ):
        super().__init__()
        self.query = query
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.collection_id = collection_id
        self.filter_text = filter_text
        self.embed_model = embed_model
        self.rewrite_settings = dict(rewrite_settings or {})
        self.rewrite_settings["groq_rewrite_model"] = rewrite_model
        self.n_results = n_results
        self.include = include
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.mode = (mode or "hybrid").lower()

    def _report_rewrite_model(self, result):
        status = f"Rewrite used {result.provider}: {result.actual_model}"
        if result.reason:
            status += f" ({result.reason})"
        self.status_update.emit(status)

    def run(self):
        rewritten_input, synonym_dict = self.query, {}
        if self.mode == "hybrid":
            rewritten_input, synonym_dict = rewrite_input_and_generate_synonyms(
                self.rewrite_settings, self.query, on_result=self._report_rewrite_model,
            )
        alpha = float(self.alpha)
        beta = float(self.beta)
        gamma = float(self.gamma)
        delta = float(self.delta)
        mode = self.mode
        ds_args = json.dumps({
            "module_path": _MODULE_PATH,
            "path": self.chroma_path,
            "col_name": self.collection_name,
            "collection_id": self.collection_id,
            "filter_text": self.filter_text,
            "query": self.query,
            "embedding_model": self.embed_model,
            "rewritten_input": rewritten_input,
            "synonym_dict": synonym_dict,
            "top_k": int(self.n_results),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "mode": mode,
            "include": self.include,
        })
        try:
            env = {**os.environ, "_DS_ARGS": ds_args, "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run(
                [sys.executable, "-c", _SEARCH_SCRIPT],
                capture_output=True,
                encoding="utf-8",
                timeout=CAPTURE_PROCESS_TIMEOUT,
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Subprocess query failed rc={result.returncode} stderr={result.stderr.strip()} stdout={result.stdout.strip()}"
                )
            payload = json.loads(result.stdout)
            formatted = payload.get("results") or []
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
            gamma, delta = 0.5, 0.5
            if hasattr(self, "statusBar"):
                self.statusBar.showMessage("Hybrid weights invalid; using lexical and phrase search", 4000)
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
        self.invalidate_direct_results()
        self.statusBar.showMessage("Searching...")
        # Run subprocess query off the UI thread
        provider = self.settings.get("generation_provider", "groq")
        rewrite_settings = {
            "generation_provider": provider,
            "strict_generation_model": self.settings.get("strict_generation_model", False),
        }
        if provider == "groq":
            rewrite_settings["generation_router_groq"] = self.get_generation_router(provider)
        self._direct_request_scope = self._direct_scope()
        path, name, collection_id, filter_text = self._direct_request_scope
        worker = DirectSearchSubprocessWorker(
            query=query,
            chroma_path=path,
            collection_name=name,
            collection_id=collection_id,
            filter_text=filter_text,
            embed_model=self.settings.get("embedding_model"),
            rewrite_model=self.settings.get("groq_rewrite_model", self.settings.get("model_name", "")),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            mode=self.settings.get("direct_search_mode", "hybrid"),
            rewrite_settings=rewrite_settings,
        )
        worker.setParent(self)
        worker.results_ready.connect(self._on_direct_worker_results)
        worker.error_occurred.connect(self._on_direct_worker_error)
        worker.status_update.connect(self._on_direct_rewrite_status)
        worker.finished.connect(worker.deleteLater)
        self._direct_worker = worker
        self._direct_rewrite_status = ""
        worker.start()

    def _direct_scope(self):
        filt = getattr(self, "direct_filter_input", None)
        path_widget = getattr(self, "chromadb_path_edit", None)
        name_widget = getattr(self, "collection_combo", None)
        path = path_widget.text() if path_widget else self.settings.get("chromadb_path")
        name = name_widget.currentText() if name_widget else self.settings.get("collection_name")
        path = canonical_store_path(path) if path else ""
        selection = persistent_selection(self.collection)
        collection_id = (
            selection["expected_id"] if selection and (path, name) == (selection["path"], selection["name"])
            else None
        )
        return (path, name, collection_id, filt.text().strip().casefold() if filt else "")

    def invalidate_direct_results(self, *_):
        self._direct_request_scope = None
        self._direct_results = []
        detail = getattr(self, "search_result_detail", None)
        if detail is not None:
            detail.clear()
        self.apply_direct_search_view()

    def _direct_request_is_current(self):
        if self.sender() is not getattr(self, "_direct_worker", None):
            return False
        return (
            self._direct_request_scope is not None
            and self._direct_request_scope == self._direct_scope()
        )

    def apply_direct_search_view(self):
        view = list(getattr(self, "_direct_results", []) or [])
        sort_combo = getattr(self, "direct_sort_combo", None)
        if sort_combo:
            mode = sort_combo.currentText()
            if mode in {"Distance ↑", "Distance ↓"}:
                view.sort(key=lambda row: distance_key(row.get("distance"), descending=mode == "Distance ↓"))
            elif mode == "File A→Z":
                view.sort(key=lambda row: row.get("file_name", ""))
        self.on_search_results(view)

    def on_search_results(self, results):
        print(
            "[DirectSearchMixin.on_search_results] received %s results type=%s" % (len(results), type(results)),
            flush=True,
        )
        result_list = getattr(self, "search_results_list", None)
        result_table = getattr(self, "search_results_table", None)
        if result_list is not None:
            result_list.clear()
        elif result_table is not None:
            result_table.setRowCount(len(results))

        for i, res in enumerate(results):
            file_name = res.get("file_name", "Unknown")
            sim = res.get("similarity")
            dist = res.get("distance")
            sim_display = f"{sim:.4f}" if isinstance(sim, (int, float)) else "n/a"
            dist_display = f"{dist:.4f}" if isinstance(dist, (int, float)) else "n/a"
            doc = res.get("document", "")
            preview = (doc or "")[:180].replace("\n", " ")
            line = f"{file_name}\n  sim={sim_display} dist={dist_display}\n  {preview}"
            if result_list is not None:
                item = QListWidgetItem(line)
                item.setData(Qt.UserRole, res)
                result_list.addItem(item)
            elif result_table is not None:
                name_item = QTableWidgetItem(file_name)
                name_item.setData(Qt.UserRole, res)
                result_table.setItem(i, 0, name_item)
                result_table.setItem(i, 1, QTableWidgetItem(sim_display))
                result_table.setItem(i, 2, QTableWidgetItem(dist_display))
                result_table.setItem(i, 3, QTableWidgetItem(preview))

        self.statusBar.showMessage(f"Found {len(results)} results", 3000)
        print("[DirectSearchMixin.on_search_results] table updated", flush=True)

    def _on_direct_worker_error(self, message):
        if self._direct_request_is_current():
            self.on_error(message)

    def _on_direct_rewrite_status(self, status):
        if not self._direct_request_is_current():
            return
        self._direct_rewrite_status = status
        self.statusBar.showMessage(status, 6000)

    def _on_direct_worker_results(self, results):
        if not DirectSearchMixin._direct_request_is_current(self):
            return
        self._direct_results = results
        self.apply_direct_search_view()
        status = f"Found {len(results)} results"
        if self._direct_rewrite_status:
            status += f" · {self._direct_rewrite_status}"
        self.statusBar.showMessage(status, 6000)

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
        ds_args = json.dumps({
            "module_path": _MODULE_PATH,
            "path": self.settings.get("chromadb_path"),
            "col_name": getattr(self.collection, "name", self.settings.get("collection_name")),
            "collection_id": str(self.collection.id) if hasattr(self.collection, "id") else None,
            "query": query,
            "embedding_model": self.settings.get("embedding_model"),
            "top_k": int(n_results),
            "mode": "self_test",
            "include": include,
        })
        try:
            result = subprocess.run(
                [sys.executable, "-c", _SEARCH_SCRIPT],
                env={**os.environ, "_DS_ARGS": ds_args},
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
