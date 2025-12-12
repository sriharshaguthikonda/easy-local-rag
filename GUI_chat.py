import re

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

        try:
            context_results = self.get_relevant_context_hybrid(user_input)
            print(f"[_do_send] Retrieved {len(context_results)} context results")
            self.on_context_ready(context_results)
            self.set_progress_stage("llm")
        except Exception as e:
            print(f"[_do_send] Context retrieval failed: {e}")
            self.on_error(f"Context retrieval failed: {e}")
            return

        if self.settings["just_search"]:
            print("[_do_send] just_search=True, completing without LLM")
            self.on_response_complete("")
            self.set_progress_stage("done")
            return

        # Start worker for LLM calls only
        print("[_do_send] Creating ChatWorker...")
        self.chat_worker = ChatWorker(
            user_input, self.settings.copy(), context_results, self.conversation_history
        )
        print("[_do_send] Connecting ChatWorker signals")
        self.chat_worker.response_chunk.connect(self.on_response_chunk)
        self.chat_worker.response_complete.connect(self.on_response_complete)
        self.chat_worker.error_occurred.connect(self.on_error)
        self.chat_worker.status_update.connect(self.on_status_update)
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
            # Split into sentences for TTS and store them
            sentences = re.split(r"[.!?]+", response)
            self._tts_sentences = [s.strip() for s in sentences[:3] if s.strip()]
            
            # Start the first TTS worker if there are any sentences
            if self._tts_sentences:
                sentence = self._tts_sentences.pop(0)
                self._start_tts_worker(sentence)
                
                # If there's another sentence, prepare it in advance
                if self._tts_sentences:
                    next_sentence = self._tts_sentences.pop(0).strip()
                    if next_sentence:
                        self._next_tts_worker = TTSWorker(next_sentence, self.settings)
                        self._next_tts_worker.finished.connect(
                            lambda w=self._next_tts_worker: self._cleanup_tts_worker(w)
                        )

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

            # Highlight keywords
            keywords = res.get("keywords", [])
            for kw in keywords:
                doc = re.sub(
                    rf"\b({re.escape(kw)})\b",
                    r'<span style="background-color: #FFD700; color: black;">\1</span>',
                    doc,
                    flags=re.IGNORECASE,
                )

            context_html += f"""
            <div style="margin: 10px 0; padding: 10px; border: 1px solid #444; border-radius: 5px;">
                <b>#{i}</b> - Score: {score:.3f}<br>
                <b>File:</b> <code>{file_name}</code><br>
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
        self.set_progress_stage("error")

        self.append_message("System", f"Error: {error}", "#E74C3C")
        self.statusBar.showMessage(f"Error: {error}", 5000)

    def on_status_update(self, status):
        self.statusBar.showMessage(status)

    def _cleanup_tts_worker(self, worker):
        """Clean up finished TTS worker and start the next one if available"""
        if worker in self.tts_workers:
            self.tts_workers.remove(worker)
            worker.deleteLater()
            
            # If we have a preprocessed next worker, start it
            if hasattr(self, '_next_tts_worker') and self._next_tts_worker:
                self.tts_workers.append(self._next_tts_worker)
                self._next_tts_worker.start()
                self._next_tts_worker = None
                
                # Preprocess the next sentence if available
                self._prepare_next_tts()
    
    def _start_tts_worker(self, sentence):
        """Helper method to start a single TTS worker"""
        tts_worker = TTSWorker(sentence, self.settings)
        tts_worker.finished.connect(
            lambda w=tts_worker: self._cleanup_tts_worker(w)
        )
        self.tts_workers.append(tts_worker)
        tts_worker.start()
        
        # Prepare the next TTS worker if available
        self._prepare_next_tts()
        
    def _prepare_next_tts(self):
        """Prepare the next TTS worker in advance if sentences are available"""
        if hasattr(self, '_tts_sentences') and self._tts_sentences and not hasattr(self, '_next_tts_worker'):
            sentence = self._tts_sentences.pop(0).strip()
            if sentence:
                self._next_tts_worker = TTSWorker(sentence, self.settings)
                self._next_tts_worker.finished.connect(
                    lambda w=self._next_tts_worker: self._cleanup_tts_worker(w)
                )
                # Don't start it yet, just prepare it
