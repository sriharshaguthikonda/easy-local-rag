import json
import subprocess
import sys

from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem
from PyQt5.QtCore import Qt, QTimer

import ollama
from GUI_workers import ChromaDBSearchWorker


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
        self.update_settings_from_ui()
        print("[DirectSearchMixin.direct_chromadb_search] settings embedding_model=%s" % self.settings.get("embedding_model"), flush=True)
        self.statusBar.showMessage("Searching...")
        try:
            res = self._query_via_subprocess(
                query,
                n_results=5,
                include=["documents", "metadatas", "distances"],
            )
            formatted = []
            docs = res.get("documents", [[]])
            metas = res.get("metadatas", [[]])
            dists = res.get("distances", [[]])
            for meta, doc, dist in zip(metas[0], docs[0], dists[0]):
                formatted.append(
                    {
                        "file_name": meta.get("file_name", "Unknown") if isinstance(meta, dict) else "Unknown",
                        "document": doc,
                        "distance": dist,
                        "similarity": 1.0 - dist if isinstance(dist, (int, float)) else None,
                        "metadata": meta,
                    }
                )
            print("[DirectSearchMixin.direct_chromadb_search] subprocess returned %s results" % len(formatted), flush=True)
            self.on_search_results(formatted)
        except Exception as e:
            import traceback as _tb

            print("[DirectSearchMixin.direct_chromadb_search] subprocess path error: %s" % e, flush=True)
            _tb.print_exc()
            self.on_error(str(e))

    def on_search_results(self, results):
        print(
            "[DirectSearchMixin.on_search_results] received %s results type=%s" % (len(results), type(results)),
            flush=True,
        )
        self.search_results_table.setRowCount(len(results))

        for i, res in enumerate(results):
            print(
                "[DirectSearchMixin.on_search_results] row=%s file=%s sim=%.4f distance=%.4f"
                % (i, res.get("file_name"), res.get("similarity"), res.get("distance")),
                flush=True,
            )
            self.search_results_table.setItem(i, 0, QTableWidgetItem(res["file_name"]))
            self.search_results_table.setItem(i, 1, QTableWidgetItem(f"{res['similarity']:.4f}"))
            self.search_results_table.setItem(i, 2, QTableWidgetItem(res["document"][:100]))

            # Store full data
            self.search_results_table.item(i, 0).setData(Qt.UserRole, res)

        self.statusBar.showMessage(f"Found {len(results)} results", 3000)
        print("[DirectSearchMixin.on_search_results] table updated", flush=True)

    def _on_search_worker_finished(self):
        print("[DirectSearchMixin._on_search_worker_finished] search worker finished", flush=True)

    def _on_search_worker_destroyed(self, obj):
        print("[DirectSearchMixin._on_search_worker_destroyed] worker destroyed: %s" % obj, flush=True)

    def show_search_result_detail(self, item):
        print("[DirectSearchMixin.show_search_result_detail] invoked", flush=True)
        row = item.row()
        data = self.search_results_table.item(row, 0).data(Qt.UserRole)

        if data:
            print(
                "[DirectSearchMixin.show_search_result_detail] row=%s file=%s distance=%.4f"
                % (row, data.get("file_name"), data.get("distance")),
                flush=True,
            )
            detail = f"""
File: {data['file_name']}
Similarity: {data['similarity']:.4f}
Distance: {data['distance']:.4f}

--- Document Content ---
{data['document']}

--- Metadata ---
{json.dumps(data['metadata'], indent=2, default=str)}
"""
            QMessageBox.information(self, "Search Result Detail", detail)

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
