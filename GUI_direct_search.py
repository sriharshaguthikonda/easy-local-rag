import json

from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem
from PyQt5.QtCore import Qt

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
        self.update_settings_from_ui()
        print("[DirectSearchMixin.direct_chromadb_search] settings embedding_model=%s" % self.settings.get("embedding_model"), flush=True)
        self.statusBar.showMessage("Searching...")

        self.search_worker = ChromaDBSearchWorker(
            query,
            self.collection,
            self.settings["embedding_model"],
            n_results=30,
        )
        print("[DirectSearchMixin.direct_chromadb_search] worker created, connecting signals and starting", flush=True)
        self.search_worker.results_ready.connect(self.on_search_results)
        self.search_worker.error_occurred.connect(self.on_error)
        self.search_worker.start()
        print("[DirectSearchMixin.direct_chromadb_search] worker started", flush=True)

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
            QMessageBox.information(self, "Document Details", detail)
