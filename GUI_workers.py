import os
import re
import json
import time
import threading

from PyQt5.QtCore import QThread, pyqtSignal

import ollama
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

from rag_prompting import (
    AnswerPreparationError, AnswerResult, AnswerStatus, build_numbered_sources,
    generate_structural_answer, prepare_structural_answer,
)

from embedding_contract import validate_embedding_compatibility


class ChatWorker(QThread):
    """Generate one answer through the selected provider router.

    Retrieval happens before this thread starts.  The worker owns only the
    prepared messages and generation stream, which keeps retries inside the
    router and prevents duplicate evidence or assistant history entries.
    """

    response_chunk = pyqtSignal(str)
    response_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)
    generation_result = pyqtSignal(object)

    def __init__(
        self,
        user_input,
        settings,
        context_results,
        conversation_history,
        router=None,
    ):
        super().__init__()
        self.user_input = user_input
        self.settings = settings
        self.context_results = context_results  # Pre-fetched context from main thread
        self.conversation_history = conversation_history
        provider = settings.get("generation_provider", "groq")
        selected = router or settings.get(f"generation_router_{provider}")
        if provider == "lmstudio":
            self.router = selected if getattr(selected, "provider", None) == provider else None
        else:
            self.router = selected or settings.get("generation_router")
        self._cancel_event = threading.Event()
        self._publication_lock = threading.Lock()
        self._published = False
        self.groq_client = settings.get("groq_client")
        print(
            "[ChatWorker.__init__] created worker id=%s, thread=%s"
            % (id(self), threading.get_ident())
        )

    def request_cancel(self):
        """Request cooperative cancellation at the next router boundary."""
        with self._publication_lock:
            if self._published:
                return False
            self._cancel_event.set()
            self.requestInterruption()
            return True

    def is_cancelled(self):
        return self._cancel_event.is_set() or self.isInterruptionRequested()

    def _router_from_settings(self):
        if self.router is not None:
            return self.router
        provider = self.settings.get("generation_provider", "groq")
        try:
            import generation_models

            if provider == "ollama":
                self.router = generation_models.ollama_router(
                    client=self.settings.get("ollama_client"),
                    host=self.settings.get("ollama_host"),
                )
            elif provider == "lmstudio":
                self.router = generation_models.lmstudio_router()
            elif provider == "groq":
                self.router = generation_models.groq_router(
                    client=self.settings.get("groq_client")
                )
            return self.router
        except ImportError:
            return None

    def _preferred_model(self):
        provider = self.settings.get("generation_provider", "groq")
        return self.settings.get(f"{provider}_model")

    def _router_generate(self, prepared):
        router = self._router_from_settings()
        provider = self.settings.get("generation_provider", "groq")
        preferred = self._preferred_model()
        max_output = int(self.settings.get("generation_max_output_tokens", 1024))
        options = dict(self.settings.get("generation_options", {}) or {})
        options.pop("text_only", None)
        if provider == "lmstudio":
            options.setdefault("mode", "local-chat")
        else:
            options.setdefault("temperature", 1)
            options.setdefault("top_p", 1)
        if provider == "groq":
            options.setdefault("service_tier", "on_demand")
        elif provider == "ollama":
            options.setdefault("keep_alive", -1)

        return generate_structural_answer(
            router, prepared, provider=provider, preferred=preferred,
            strict=bool(self.settings.get("strict_generation_model", False)),
            max_output_tokens=max_output, cancelled=self.is_cancelled, **options,
        )

    def run(self):
        if self.settings.get("just_search", False):
            self.response_complete.emit("")
            return

        provider = self.settings.get("generation_provider", "groq")
        prepared = None
        try:
            if self.is_cancelled():
                result = AnswerResult(AnswerStatus.CANCELLED, provider=provider)
            else:
                self.status_update.emit("Generating response...")
                sources = build_numbered_sources(self.context_results)
                prepared = prepare_structural_answer(
                    self.settings["system_message"], self.user_input, sources,
                    self.conversation_history,
                )
                result = self._router_generate(prepared)
        except AnswerPreparationError as error:
            result = AnswerResult(error.status, provider=provider)
        except ValueError:
            status = AnswerStatus.INSUFFICIENT_EVIDENCE if prepared is None else AnswerStatus.PROVIDER_FAILED
            result = AnswerResult(status, prepared.evidence if prepared else None, provider=provider)
        except Exception:
            result = AnswerResult(
                AnswerStatus.PROVIDER_FAILED, prepared.evidence if prepared else None, provider=provider,
            )

        with self._publication_lock:
            if self.is_cancelled():
                result = AnswerResult(AnswerStatus.CANCELLED, result.evidence, provider=provider)
            if result.accepted:
                self.conversation_history.extend([
                    {"role": "user", "content": self.user_input},
                    result.assistant_turn(),
                ])
            self._published = True

        if result.accepted:
            status = f"Answered with {result.actual_model}"
            if result.reason:
                status += f" ({result.reason})"
            self.status_update.emit(status)
            self.response_chunk.emit(result.text)
        self.generation_result.emit(result)
        self.response_complete.emit(result.text)
        if result.status is AnswerStatus.CANCELLED:
            self.status_update.emit(result.message)
        elif not result.accepted:
            self.error_occurred.emit(result.message)


class TTSWorker(QThread):
    """Worker thread for TTS playback"""

    finished = pyqtSignal()

    def __init__(self, text, settings):
        super().__init__()
        self.text = text
        self.settings = settings

    def run(self):
        try:
            print(f"[TTSWorker.run] worker id={id(self)} starting")
            if not self.text.strip():
                print("[TTSWorker.run] empty text, exiting early")
                return

            tts = gTTS(
                text=self.text,
                lang=self.settings.get("tts_lang", "en"),
                tld=self.settings.get("tts_tld", "co.uk"),
            )
            from io import BytesIO

            audio_fp = BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)

            audio = AudioSegment.from_file(audio_fp, format="mp3")
            audio = audio.speedup(
                playback_speed=self.settings.get("tts_speed", 1.4)
            )
            audio = audio + (self.settings.get("tts_volume", 0.5) * 10)

            play(audio)
        except Exception as e:
            print(f"[TTSWorker.run] ERROR: {e}")
        finally:
            print(
                f"[TTSWorker.run] finishing worker id={id(self)}, isRunning={self.isRunning()}, isFinished={self.isFinished()}"
            )
            self.finished.emit()


class ChromaDBSearchWorker(QThread):
    """Worker for direct ChromaDB vector search"""

    results_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, query, collection, embedding_model, n_results=20):
        super().__init__()
        self.query = query
        self.collection = collection
        self.embedding_model = embedding_model
        self.n_results = n_results
        print(
            "[ChromaDBSearchWorker.__init__] query_len=%s collection=%s model=%s n_results=%s"
            % (len(query), getattr(collection, "name", "<no-name>"), embedding_model, n_results),
            flush=True,
        )

    def run(self):
        try:
            if not validate_embedding_compatibility(self.collection, self.embedding_model):
                self.results_ready.emit([])
                return

            thread_id = threading.get_ident()
            print(
                "[ChromaDBSearchWorker.run] thread=%s starting embedding call" % thread_id,
                flush=True,
            )
            embedding = ollama.embeddings(
                model=self.embedding_model,
                prompt=self.query,
                keep_alive=-1,
            )["embedding"]
            print(
                "[ChromaDBSearchWorker.run] querying collection=%s n_results=%s thread=%s"
                % (getattr(self.collection, "name", "<no-name>"), self.n_results, thread_id),
                flush=True,
            )
            try:
                start = time.time()
                # Minimize payload for debugging stability; can re-enable docs/metadatas later
                results = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=self.n_results,
                    include=["distances"],
                )
                elapsed = time.time() - start
                print(
                    "[ChromaDBSearchWorker.run] query() finished in %.3fs thread=%s" % (elapsed, thread_id),
                    flush=True,
                )
            except Exception as qe:
                import traceback

                print("[ChromaDBSearchWorker.run] query() raised: %s (%s)" % (qe, type(qe)), flush=True)
                traceback.print_exc()
                self.error_occurred.emit(f"Chroma query failed: {qe}")
                return

            if results is None:
                print("[ChromaDBSearchWorker.run] results is None", flush=True)
                self.error_occurred.emit("Chroma query returned None")
                return

            print(
                "[ChromaDBSearchWorker.run] raw results keys=%s lens=%s"
                % (
                    list(results.keys()),
                    {k: (len(v[0]) if isinstance(v, list) and v else "n/a") for k, v in results.items()},
                ),
                flush=True,
            )

            # Defensive checks on expected structure
            try:
                metadatas = results.get("metadatas", [])
                documents = results.get("documents", [])
                distances = results.get("distances", [])
                print(
                    "[ChromaDBSearchWorker.run] unpack lengths metas=%s docs=%s dists=%s"
                    % (len(metadatas), len(documents), len(distances)),
                    flush=True,
                )
                first_meta_len = len(metadatas[0]) if metadatas and metadatas[0] else 0
                first_doc_len = len(documents[0]) if documents and documents[0] else 0
                first_dist_len = len(distances[0]) if distances and distances[0] else 0
                print(
                    "[ChromaDBSearchWorker.run] first batch lens metas=%s docs=%s dists=%s"
                    % (first_meta_len, first_doc_len, first_dist_len),
                    flush=True,
                )
            except Exception as se:
                import traceback

                print("[ChromaDBSearchWorker.run] structure check failed: %s" % se, flush=True)
                traceback.print_exc()
                self.error_occurred.emit(f"Chroma result structure error: {se}")
                return

            formatted = []
            try:
                for idx, (meta, doc, dist) in enumerate(
                    zip(
                        results["metadatas"][0],
                        results["documents"][0],
                        results["distances"][0],
                    )
                ):
                    formatted.append(
                        {
                            "file_name": meta.get("file_name", "Unknown"),
                            "document": doc,
                            "distance": dist,
                            "similarity": 1.0 - dist,
                            "metadata": meta,
                        }
                    )
                print("[ChromaDBSearchWorker.run] emitting %s formatted results" % len(formatted), flush=True)
                self.results_ready.emit(formatted)
            except Exception as pe:
                import traceback

                print("[ChromaDBSearchWorker.run] post-process error: %s" % pe, flush=True)
                traceback.print_exc()
                self.error_occurred.emit(f"Chroma post-process failed: {pe}")
        except Exception as e:
            import traceback

            print("[ChromaDBSearchWorker.run] error: %s" % e, flush=True)
            traceback.print_exc()
            self.error_occurred.emit(str(e))
        finally:
            print(
                "[ChromaDBSearchWorker.run] exit thread=%s isRunning=%s isFinished=%s"
                % (threading.get_ident(), self.isRunning(), self.isFinished()),
                flush=True,
            )
