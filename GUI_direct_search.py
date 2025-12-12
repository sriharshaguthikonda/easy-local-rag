import json
import os
import subprocess
import sys

from PyQt5.QtWidgets import QMessageBox, QListWidgetItem
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class DirectSearchSubprocessWorker(QThread):
    results_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, query, chroma_path, collection_name, embed_model, n_results, include):
        super().__init__()
        self.query = query
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.n_results = n_results
        self.include = include

    def run(self):
        include_literal = "[" + ",".join(repr(x) for x in self.include) + "]"
        script = f"""
import json, chromadb, ollama
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
path = {repr(self.chroma_path)}
col_name = {repr(self.collection_name)}
query = {repr(self.query)}
emb = ollama.embeddings(model={repr(self.embed_model)}, prompt=query, keep_alive=-1)["embedding"]
client = chromadb.PersistentClient(path=path, settings=Settings(), tenant=DEFAULT_TENANT, database=DEFAULT_DATABASE)
col = client.get_collection(col_name)
res = col.query(query_embeddings=[emb], n_results={int(self.n_results)}, include={include_literal})
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
                raise RuntimeError(
                    f"Subprocess query failed rc={result.returncode} stderr={result.stderr.strip()} stdout={result.stdout.strip()}"
                )
            if not result.stdout:
                raise RuntimeError("Subprocess query produced no output")
            res = json.loads(result.stdout)
            docs = res.get("documents", [[]])
            metas = res.get("metadatas", [[]])
            dists = res.get("distances", [[]])
            formatted = []
            for meta, doc, dist in zip(metas[0], docs[0], dists[0]):
                formatted.append(
                    {
                        "file_name": meta.get("file_name", "Unknown") if isinstance(meta, dict) else "Unknown",
                        "file_path": meta.get("file_path") if isinstance(meta, dict) else None,
                        "document": doc,
                        "distance": dist,
                        "similarity": 1.0 - dist if isinstance(dist, (int, float)) else None,
                        "metadata": meta,
                    }
                )
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
        print(
            "[DirectSearchMixin.direct_chromadb_search] settings embedding_model=%s top_k=%s"
            % (self.settings.get("embedding_model"), top_k),
            flush=True,
        )
        self.statusBar.showMessage("Searching...")
        # Run subprocess query off the UI thread
        self._direct_worker = DirectSearchSubprocessWorker(
            query=query,
            chroma_path=self.settings.get("chromadb_path"),
            collection_name=getattr(self.collection, "name", self.settings.get("collection_name")),
            embed_model=self.settings.get("embedding_model"),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
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
            doc = res.get("document", "")
            preview = (doc or "")[:180].replace("\n", " ")
            line = f"{file_name}\n  sim={sim:.4f} dist={dist:.4f}\n  {preview}"
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

    def open_search_result_file(self, item):
        if not item:
            return
        data = item.data(Qt.UserRole) or {}
        path = data.get("file_path") or data.get("file_name")
        if not path:
            QMessageBox.information(self, "Open File", "No file path available.")
            return
        if not os.path.exists(path):
            QMessageBox.warning(self, "Open File", f"File not found:\n{path}")
            return
        try:
            os.startfile(path)
        except Exception as e:
            QMessageBox.warning(self, "Open File", f"Failed to open file:\n{path}\n\nError: {e}")

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
