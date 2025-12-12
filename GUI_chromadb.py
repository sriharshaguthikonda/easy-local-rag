from PyQt5.QtWidgets import QFileDialog, QMessageBox

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings


class ChromaDBMixin:
    """ChromaDB-related helpers split out of the main GUI class."""

    def browse_chromadb_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select ChromaDB Directory")
        if path:
            self.chromadb_path_edit.setText(path)

    def connect_chromadb(self, show_errors=True):
        try:
            self.update_settings_from_ui()
            path = self.settings["chromadb_path"]

            print(f"    Connecting to ChromaDB at: {path}")
            self.chromadb_client = chromadb.PersistentClient(
                path=path,
                settings=Settings(),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
            print("    ChromaDB client created")

            self.refresh_collections()
            print("    Collections refreshed")

            count = self.collection_combo.count()
            print(f"    Collection count: {count}")
            if count > 0:
                collection_name = self.collection_combo.currentText()
                print(f"    Getting collection: {collection_name}")
                self.collection = self.chromadb_client.get_collection(collection_name)
                print("    Got collection, updating status bar")
                self.statusBar.showMessage(f"Connected to collection: {collection_name}", 5000)
                print("    Skipping refresh_stats for now")
            else:
                self.statusBar.showMessage("Connected but no collections found", 5000)
            print("    connect_chromadb complete")

        except Exception as e:
            print(f"    ChromaDB connection error: {e}")
            self.statusBar.showMessage(f"Connection failed: {e}", 5000)
            if show_errors and self.isVisible():
                QMessageBox.critical(self, "Connection Error", f"Failed to connect to ChromaDB:\n{e}")

    def refresh_collections(self):
        if self.chromadb_client:
            try:
                collections = self.chromadb_client.list_collections()
                self.collection_combo.clear()
                for col in collections:
                    self.collection_combo.addItem(col.name)

                idx = self.collection_combo.findText(self.settings["collection_name"])
                if idx >= 0:
                    self.collection_combo.setCurrentIndex(idx)
            except Exception as e:
                print(f"Error refreshing collections: {e}")

    def refresh_stats(self):
        print("      refresh_stats called")
        if self.collection:
            try:
                print("      Getting count...")
                count = self.collection.count()
                print(f"      Count: {count}")
                stats_text = f"""**Collection Statistics**

Collection Name: {self.collection.name}
Total Documents: {count}

Current Settings:
- Embedding Model: {self.settings['embedding_model']}
- Groq Model: {self.settings['groq_model']}
- Top K: {self.settings['top_k']}
"""
                print("      Setting text...")
                self.stats_display.setPlainText(stats_text)
                print("      Text set")
            except Exception as e:
                print(f"      Error in refresh_stats: {e}")
                self.stats_display.setText(f"Error getting stats: {e}")
