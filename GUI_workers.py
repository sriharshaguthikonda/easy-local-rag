import os
import re
import json
import threading

from PyQt5.QtCore import QThread, pyqtSignal

import ollama
from groq import Groq
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play


class ChatWorker(QThread):
    """Worker thread for chat operations - only handles LLM calls, not ChromaDB"""
    response_chunk = pyqtSignal(str)
    response_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, user_input, settings, context_results, conversation_history):
        super().__init__()
        self.user_input = user_input
        self.settings = settings
        self.context_results = context_results  # Pre-fetched context from main thread
        self.conversation_history = conversation_history
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print(
            "[ChatWorker.__init__] created worker id=%s, thread=%s",
            id(self),
            threading.get_ident(),
        )

    def run(self):
        try:
            print(
                "[ChatWorker.run] starting worker id=%s on thread=%s",
                id(self),
                threading.get_ident(),
            )
            self.status_update.emit("Generating response...")

            if self.settings.get("just_search", False):
                print("[ChatWorker.run] just_search=True, returning early")
                self.response_complete.emit("")
                return

            relevant_context = "\n\n".join(
                [res.get("document", "") for res in self.context_results]
            )
            print(
                f"[ChatWorker.run] Context docs: {len(self.context_results)}; length={len(relevant_context)}"
            )

            if relevant_context:
                user_input_with_context = f"{relevant_context}\n\n{self.user_input}"
            else:
                user_input_with_context = self.user_input

            self.conversation_history.append(
                {"role": "user", "content": user_input_with_context}
            )

            messages = [
                {"role": "system", "content": self.settings["system_message"]},
                *self.conversation_history,
            ]

            print(f"[ChatWorker.run] Calling Groq with model={self.settings['groq_model']}")
            response = self.groq_chat(messages)
            print(f"[ChatWorker.run] Groq response length={len(response)}")
            self.conversation_history.append({"role": "assistant", "content": response})
            self.response_complete.emit(response)
            print(
                "[ChatWorker.run] finished worker id=%s on thread=%s",
                id(self),
                threading.get_ident(),
            )

        except Exception as e:
            print(f"[ChatWorker.run] ERROR: {e}")
            import traceback as _tb
            _tb.print_exc()
            self.status_update.emit("Error during generation")
            self.error_occurred.emit(str(e))
        finally:
            print(
                "[ChatWorker.run] exiting run() worker id=%s, isRunning=%s, isFinished=%s",
                id(self),
                self.isRunning(),
                self.isFinished(),
            )

    def groq_chat(self, messages):
        try:
            print("[ChatWorker.groq_chat] starting streaming call")
            stream = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.settings["groq_model"],
                temperature=1,
                max_tokens=4096,
                top_p=1,
                stream=True,
            )

            response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunk_text = chunk.choices[0].delta.content
                    response += chunk_text
                    self.response_chunk.emit(chunk_text)

            return response
        except Exception as e:
            print(f"[ChatWorker.groq_chat] error: {e}")
            self.status_update.emit("Falling back to Ollama...")
            return self.ollama_chat(messages)

    def ollama_chat(self, messages):
        print("[ChatWorker.ollama_chat] starting Ollama fallback")
        stream = ollama.chat(
            model=self.settings["ollama_model"],
            messages=messages,
            stream=True,
            keep_alive=-1,
        )

        response = ""
        for chunk in stream:
            chunk_text = chunk["message"]["content"]
            response += chunk_text
            self.response_chunk.emit(chunk_text)

        return response


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

    def run(self):
        try:
            embedding = ollama.embeddings(
                model=self.embedding_model,
                prompt=self.query,
                keep_alive=-1,
            )["embedding"]

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=self.n_results,
                include=["documents", "metadatas", "distances"],
            )

            formatted = []
            for meta, doc, dist in zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0],
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

            self.results_ready.emit(formatted)
        except Exception as e:
            self.error_occurred.emit(str(e))
