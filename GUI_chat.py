import re
from html import escape

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QTextCursor

from GUI_workers import ChatWorker, TTSWorker


class ChatFunctionalityMixin:
    """Encapsulates chat send/search workflows shared by the main GUI."""

    # =========================================================================
    # CHAT FUNCTIONALITY
    # =========================================================================

    def send_message(self):
        self.settings["just_search"] = False
        self._do_send()

    def search_only(self):
        self.settings["just_search"] = True
        self._do_send()

    def _do_send(self):
        user_input = self.chat_input.toPlainText().strip()
        print(f"[_do_send] Called with input length={len(user_input)}")
        if not user_input:
            print("[_do_send] Empty input, aborting")
            return

        if not self.collection:
            print("[_do_send] No collection connected")
            QMessageBox.warning(self, "Error", "Please connect to ChromaDB first!")
            return

        self.update_settings_from_ui()

        # Display user message
        print("[_do_send] Appending user message to chat display")
        self.append_message("You", user_input, "#4A90D9")
        self.chat_input.clear()

        # Disable buttons and show progress
        self.send_btn.setEnabled(False)
        self.search_only_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._streaming_started = False
        self.set_progress_stage("context")

        # Get context in main thread (ChromaDB is not thread-safe)
        print("[_do_send] Retrieving context in main thread...")
        self.statusBar.showMessage("Retrieving context...")
        QApplication.processEvents()  # Update UI

        # Search Only is deliberately lexical and provider-free.  It must not
        # invoke the rewrite/embedding path, create a router, or leave a
        # failed generation worker behind.
        if self.settings["just_search"]:
            try:
                from offline_search import search_documents

                sources = search_documents(
                    self.collection,
                    user_input,
                    int(self.settings.get("top_k", 5)),
                )
                context_results = []
                for source in sources or []:
                    metadata = source.get("metadata") or {}
                    context_results.append(
                        {
                            "id": source.get("id"),
                            "collection_id": source.get("collection_id"),
                            "chunk_id": source.get("chunk_id"),
                            "meta": metadata,
                            "metadata": metadata,
                            "document": source.get("document", ""),
                            "final_score": source.get("relevance_score", 0.0),
                            "keywords": [],
                        }
                    )
                self._last_offline_search_sources = list(sources or [])
                print(
                    "[_do_send] offline search returned %s results"
                    % len(context_results)
                )
                self.on_context_ready(context_results)
                self.statusBar.showMessage(
                    f"Offline search found {len(context_results)} results", 3000
                )
                self.on_response_complete("")
                if hasattr(self, "set_progress_stage"):
                    self.set_progress_stage("done")
            except Exception as error:
                self.on_error(f"Offline search failed: {error}")
            return

        try:
            context_results = self.get_relevant_context_hybrid(user_input)
            print(f"[_do_send] Retrieved {len(context_results)} context results")
            self.on_context_ready(context_results)
            self.set_progress_stage("llm")
        except Exception as e:
            print(f"[_do_send] Context retrieval failed: {e}")
            self.on_error(f"Context retrieval failed: {e}")
            return

        # Start worker for LLM calls only
        print("[_do_send] Creating ChatWorker...")
        provider = self.settings.get("generation_provider", "groq")
        router = None
        get_router = getattr(self, "get_generation_router", None)
        if callable(get_router):
            router = get_router(provider)
        self.chat_worker = ChatWorker(
            user_input,
            self.settings.copy(),
            context_results,
            self.conversation_history,
            router=router,
        )
        print("[_do_send] Connecting ChatWorker signals")
        self.chat_worker.response_chunk.connect(self.on_response_chunk)
        self.chat_worker.response_complete.connect(self.on_response_complete)
        self.chat_worker.error_occurred.connect(self.on_error)
        self.chat_worker.status_update.connect(self.on_status_update)
        if hasattr(self.chat_worker, "generation_result"):
            self.chat_worker.generation_result.connect(self.on_generation_result)
        print("[_do_send] Starting ChatWorker thread")
        self.chat_worker.start()
        self.set_progress_stage("streaming")

        # Add assistant placeholder
        self.append_message("Assistant", "", "#2ECC71", start_only=True)

    def on_response_chunk(self, chunk):
        if not getattr(self, "_streaming_started", False):
            self._streaming_started = True
            self.set_progress_stage("streaming")
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(chunk)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def on_response_complete(self, response):
        print(
            f"[on_response_complete] Called, response length={len(response) if response else 0}"
        )
        self.send_btn.setEnabled(True)
        self.search_only_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if hasattr(self, "set_progress_stage"):
            self.set_progress_stage("done")
        self.statusBar.showMessage("Ready", 3000)

        # Clean up worker thread (it has already finished when signal is emitted)
        if self.chat_worker:
            print("[on_response_complete] Cleaning up ChatWorker")
            if self.chat_worker.isRunning():
                self.chat_worker.wait(2000)
            self.chat_worker.deleteLater()
            self.chat_worker = None

        if response and self.tts_enabled:
            sentences = re.split(r"[.!?]+", response)
            self._tts_sentences = [s.strip() for s in sentences[:3] if s.strip()]
            self._play_next_tts_sentence()

        # Add newlines after response
        self.chat_display.append("\n")

    def on_context_ready(self, results):
        self.context_display.clear()

        if not results:
            self.context_display.setText("No context found.")
            return

        context_html = "<h3>📄 Retrieved Context</h3><hr>"

        for i, res in enumerate(results, 1):
            meta = res.get("meta", {})
            file_name = meta.get("file_name", "Unknown")
            score = res.get("final_score", 0)
            doc = res.get("document", "")[:500]

            keywords = [re.escape(word) for word in res.get("keywords", []) if word]
            if keywords:
                pattern = rf"\b({'|'.join(keywords)})\b"
                parts = re.split(pattern, doc, flags=re.IGNORECASE)
                doc = "".join(
                    f'<span style="background-color: #FFD700; color: black;">{escape(part)}</span>'
                    if index % 2 else escape(part)
                    for index, part in enumerate(parts)
                )
            else:
                doc = escape(doc)

            context_html += f"""
            <div style="margin: 10px 0; padding: 10px; border: 1px solid #444; border-radius: 5px;">
                <b>#{i}</b> - Score: {score:.3f}<br>
                <b>File:</b> <code>{escape(file_name)}</code><br>
                <hr style="margin: 5px 0;">
                <p style="font-size: 11px;">{doc}...</p>
            </div>
            """

        self.context_display.setHtml(context_html)

    def on_error(self, error):
        print(f"[GUI_chat.on_error] received error: {error}", flush=True)
        self.send_btn.setEnabled(True)
        self.search_only_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if hasattr(self, "set_progress_stage"):
            self.set_progress_stage("error")

        self.append_message("System", f"Error: {error}", "#E74C3C")
        self.statusBar.showMessage(f"Error: {error}", 5000)

    def on_status_update(self, status):
        self.statusBar.showMessage(status)

    def on_generation_result(self, result):
        """Display the safe selected/actual model status from the router."""
        self._last_answer_result = result
        if result is None or not result.accepted:
            return
        actual = result.actual_model
        reason = getattr(result, "reason", None)
        if actual:
            provider = getattr(result, "provider", None)
            model_label = f"{provider}: {actual}" if provider else actual
            message = f"Answered with {model_label}"
            if reason:
                message += f" ({reason})"
            self.append_message("Model", message, "#9CA3AF")

    def _cleanup_tts_worker(self, worker):
        if worker in self.tts_workers:
            self.tts_workers.remove(worker)
            worker.deleteLater()
            if self.tts_enabled:
                self._play_next_tts_sentence()

    def _play_next_tts_sentence(self):
        if not self.tts_enabled or self.tts_workers or not self._tts_sentences:
            return
        worker = TTSWorker(self._tts_sentences.pop(0), self.settings)
        worker.finished.connect(lambda: self._cleanup_tts_worker(worker))
        self.tts_workers.append(worker)
        worker.start()
