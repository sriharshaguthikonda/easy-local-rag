import json

from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem
from PyQt5.QtCore import Qt

from GUI_workers import ChromaDBSearchWorker


class DirectSearchMixin:
    """Direct ChromaDB search helpers split from main GUI."""

    def direct_chromadb_search(self):
        query = self.direct_search_input.text().strip()
        if not query or not self.collection:
            return

        self.update_settings_from_ui()
        self.statusBar.showMessage("Searching...")

        self.search_worker = ChromaDBSearchWorker(
            query,
            self.collection,
            self.settings["embedding_model"],
            n_results=30,
        )
        self.search_worker.results_ready.connect(self.on_search_results)
        self.search_worker.error_occurred.connect(self.on_error)
        self.search_worker.start()

    def on_search_results(self, results):
        self.search_results_table.setRowCount(len(results))

        for i, res in enumerate(results):
            self.search_results_table.setItem(i, 0, QTableWidgetItem(res["file_name"]))
            self.search_results_table.setItem(i, 1, QTableWidgetItem(f"{res['similarity']:.4f}"))
            self.search_results_table.setItem(i, 2, QTableWidgetItem(res["document"][:100]))

            # Store full data
            self.search_results_table.item(i, 0).setData(Qt.UserRole, res)

        self.statusBar.showMessage(f"Found {len(results)} results", 3000)

    def show_search_result_detail(self, item):
        row = item.row()
        data = self.search_results_table.item(row, 0).data(Qt.UserRole)

        if data:
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
