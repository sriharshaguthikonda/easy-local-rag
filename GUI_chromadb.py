from PyQt5.QtWidgets import QFileDialog, QMessageBox

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings


class ChromaDBMixin:
    """ChromaDB-related helpers split out of the main GUI class."""

    def browse_chromadb_path(self):
        print("[ChromaDBMixin.browse_chromadb_path] opening directory dialog", flush=True)
        path = QFileDialog.getExistingDirectory(self, "Select ChromaDB Directory")
        print(f"[ChromaDBMixin.browse_chromadb_path] selected path={path!r}", flush=True)
        if path:
            self.chromadb_path_edit.setText(path)
            print("[ChromaDBMixin.browse_chromadb_path] path set on chromadb_path_edit", flush=True)

    def connect_chromadb(self, show_errors=True):
        print("[ChromaDBMixin.connect_chromadb] start", flush=True)
        try:
            print("[ChromaDBMixin.connect_chromadb] updating settings from UI", flush=True)
            self.update_settings_from_ui()
            path = self.settings["chromadb_path"]
            print(f"[ChromaDBMixin.connect_chromadb] chromadb_path={path}", flush=True)

            self.statusBar.showMessage("Connecting to ChromaDB...")
            print("[ChromaDBMixin.connect_chromadb] creating PersistentClient", flush=True)
            self.chromadb_client = chromadb.PersistentClient(
                path=path,
                settings=Settings(),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
            print("[ChromaDBMixin.connect_chromadb] client created", flush=True)

            print("[ChromaDBMixin.connect_chromadb] refreshing collections", flush=True)
            self.refresh_collections()
            print("[ChromaDBMixin.connect_chromadb] collections refreshed", flush=True)

            count = self.collection_combo.count()
            print(f"[ChromaDBMixin.connect_chromadb] collection_combo.count={count}", flush=True)
            if count > 0:
                collection_name = self.collection_combo.currentText()
                print(f"[ChromaDBMixin.connect_chromadb] current collection name={collection_name}", flush=True)
                self.collection = self.chromadb_client.get_collection(collection_name)
                print("[ChromaDBMixin.connect_chromadb] collection object acquired", flush=True)
                self.statusBar.showMessage(f"Connected to collection: {collection_name}", 5000)
                if hasattr(self, "update_connection_status"):
                    self.update_connection_status(f"Connected: {collection_name}", "#2ECC71")
                print("[ChromaDBMixin.connect_chromadb] status bar updated", flush=True)
                # Optionally refresh stats if desired
                # print("[ChromaDBMixin.connect_chromadb] calling refresh_stats", flush=True)
                # self.refresh_stats()
            else:
                print("[ChromaDBMixin.connect_chromadb] no collections found on client", flush=True)
                self.statusBar.showMessage("Connected but no collections found", 5000)
                if hasattr(self, "update_connection_status"):
                    self.update_connection_status("Connected (no collections)", "#F39C12")

            print("[ChromaDBMixin.connect_chromadb] complete", flush=True)

        except Exception as e:
            print(f"[ChromaDBMixin.connect_chromadb] ERROR: {e}", flush=True)
            self.statusBar.showMessage(f"Connection failed: {e}", 5000)
            if hasattr(self, "update_connection_status"):
                self.update_connection_status("Disconnected", "#E74C3C")
            if show_errors and self.isVisible():
                QMessageBox.critical(self, "Connection Error", f"Failed to connect to ChromaDB:\n{e}")

    def refresh_collections(self):
        print("[ChromaDBMixin.refresh_collections] start", flush=True)
        if not getattr(self, "chromadb_client", None):
            print("[ChromaDBMixin.refresh_collections] no chromadb_client, abort", flush=True)
            return

        try:
            print("[ChromaDBMixin.refresh_collections] listing collections", flush=True)
            collections = self.chromadb_client.list_collections()
            print(f"[ChromaDBMixin.refresh_collections] found {len(collections)} collections", flush=True)

            self.collection_combo.blockSignals(True)
            self.collection_combo.clear()
            for col in collections:
                print(f"[ChromaDBMixin.refresh_collections] adding collection name={col.name}", flush=True)
                self.collection_combo.addItem(col.name)
            self.collection_combo.blockSignals(False)

            preferred = self.settings.get("collection_name", "")
            print(f"[ChromaDBMixin.refresh_collections] preferred collection from settings={preferred!r}", flush=True)
            if preferred:
                idx = self.collection_combo.findText(preferred)
                print(f"[ChromaDBMixin.refresh_collections] preferred index={idx}", flush=True)
                if idx >= 0:
                    self.collection_combo.setCurrentIndex(idx)
                    print("[ChromaDBMixin.refresh_collections] preferred collection selected", flush=True)

            print("[ChromaDBMixin.refresh_collections] done", flush=True)

        except Exception as e:
            print(f"[ChromaDBMixin.refresh_collections] ERROR: {e}", flush=True)

    def refresh_stats(self):
        print("[ChromaDBMixin.refresh_stats] start", flush=True)
        if not getattr(self, "collection", None):
            print("[ChromaDBMixin.refresh_stats] no active collection, abort", flush=True)
            self.stats_display.setText("No collection connected.")
            return

        try:
            print("[ChromaDBMixin.refresh_stats] calling collection.count()", flush=True)
            count = self.collection.count()
            print(f"[ChromaDBMixin.refresh_stats] count={count}", flush=True)

            stats_text = f"""**Collection Statistics**

Collection Name: {self.collection.name}
Total Documents: {count}

Current Settings:
- Embedding Model: {self.settings.get('embedding_model')}
- Groq Model: {self.settings.get('groq_model')}
- Top K: {self.settings.get('top_k')}
"""
            print("[ChromaDBMixin.refresh_stats] updating stats_display", flush=True)
            self.stats_display.setPlainText(stats_text)
            print("[ChromaDBMixin.refresh_stats] done", flush=True)

        except Exception as e:
            print(f"[ChromaDBMixin.refresh_stats] ERROR: {e}", flush=True)
            self.stats_display.setText(f"Error getting stats: {e}")
