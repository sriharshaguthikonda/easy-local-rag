
import sys
import os
import json
import re
import io
import threading
import queue
import asyncio
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QTabWidget, QGroupBox, QFormLayout, QSplitter,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QCheckBox,
    QSlider, QScrollArea, QFrame, QStatusBar, QProgressBar, QToolBar,
    QAction, QDockWidget, QTreeWidget, QTreeWidgetItem, QTableWidget,
    QTableWidgetItem, QHeaderView, QStyle, QStyleFactory, QInputDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QTextCursor, QTextCharFormat

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

import ollama
from groq import Groq
from dotenv import load_dotenv

from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS

import numpy as np
from rank_bm25 import BM25Okapi

import nest_asyncio
nest_asyncio.apply()

load_dotenv()


# ============================================================================
# WORKER THREADS
# ============================================================================

class ChatWorker(QThread):
    """Worker thread for chat operations"""
    response_chunk = pyqtSignal(str)
    response_complete = pyqtSignal(str)
    context_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, user_input, settings, collection, conversation_history):
        super().__init__()
        self.user_input = user_input
        self.settings = settings
        self.collection = collection
        self.conversation_history = conversation_history
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
    def run(self):
        try:
            self.status_update.emit("Retrieving context...")
            context_results = self.get_relevant_context_hybrid()
            self.context_ready.emit(context_results)
            
            if self.settings.get('just_search', False):
                self.response_complete.emit("")
                return
            
            self.status_update.emit("Generating response...")
            relevant_context = "\n\n".join([res["document"] for res in context_results])
            
            if relevant_context:
                user_input_with_context = relevant_context + "\n\n" + self.user_input
            else:
                user_input_with_context = self.user_input
            
            self.conversation_history.append({"role": "user", "content": user_input_with_context})
            
            messages = [
                {"role": "system", "content": self.settings['system_message']},
                *self.conversation_history
            ]
            
            response = self.groq_chat(messages)
            self.conversation_history.append({"role": "assistant", "content": response})
            self.response_complete.emit(response)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def groq_chat(self, messages):
        try:
            stream = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.settings['groq_model'],
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
            # Fallback to Ollama
            self.status_update.emit("Falling back to Ollama...")
            return self.ollama_chat(messages)
    
    def ollama_chat(self, messages):
        stream = ollama.chat(
            model=self.settings['ollama_model'],
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
    
    def get_relevant_context_hybrid(self):
        try:
            rewritten_input, synonym_dict = self.rewrite_input_and_generate_synonyms()
            
            input_embedding = ollama.embeddings(
                model=self.settings['embedding_model'],
                prompt=rewritten_input,
                keep_alive=-1,
            )["embedding"]
            
            search_result = self.collection.query(
                query_embeddings=[input_embedding],
                n_results=50,
                include=["documents", "metadatas", "distances"],
            )
            
            # Vector results
            vector_results = [
                {
                    "meta": meta,
                    "document": doc,
                    "vector_score": 1.0 - dist,
                }
                for meta, doc, dist in zip(
                    search_result["metadatas"][0],
                    search_result["documents"][0],
                    search_result["distances"][0],
                )
            ]
            
            # Keyword matching
            non_keywords = {"is", "a", "an", "and", "the", "of", "in", "on", "at", "by", "with", "for", "to", "from"}
            keywords = [word for word in rewritten_input.lower().split() if word not in non_keywords]
            
            for key, details in synonym_dict.items():
                keywords.append(key)
                for field in ['synonyms', 'spelling_variants', 'plural_singular', 'parts_of_speech', 'related_terms']:
                    if details.get(field):
                        keywords.extend(details[field])
            
            keywords = list(set(keywords))
            
            keyword_results = []
            for meta, doc in zip(search_result["metadatas"][0], search_result["documents"][0]):
                normalized_doc = re.sub(r"(?<=\w)-\s*(?=\w)", "", doc.lower())
                match_score = sum(
                    len(re.findall(rf"\b{re.escape(kw)}\b", normalized_doc))
                    for kw in keywords
                )
                if match_score > 0:
                    keyword_results.append({"meta": meta, "document": doc, "keyword_score": match_score})
            
            # BM25
            bm25_corpus = [doc for doc in search_result["documents"][0]]
            bm25 = BM25Okapi([doc.split() for doc in bm25_corpus])
            bm25_scores = bm25.get_scores(rewritten_input.split())
            
            bm25_results = [
                {"meta": meta, "document": doc, "bm25_score": score}
                for meta, doc, score in zip(
                    search_result["metadatas"][0],
                    search_result["documents"][0],
                    bm25_scores,
                )
            ]
            
            # Normalize and combine
            alpha = self.settings['alpha']
            beta = self.settings['beta']
            gamma = self.settings['gamma']
            
            max_vector = max([r["vector_score"] for r in vector_results], default=1)
            max_keyword = max([r["keyword_score"] for r in keyword_results], default=1)
            max_bm25 = max([r["bm25_score"] for r in bm25_results], default=1)
            
            combined = {}
            for res in vector_results:
                fn = res["meta"].get("file_name", "unknown")
                combined[fn] = {
                    "meta": res["meta"],
                    "document": res["document"],
                    "final_score": alpha * (res["vector_score"] / max_vector),
                    "keywords": keywords
                }
            
            for res in keyword_results:
                fn = res["meta"].get("file_name", "unknown")
                if fn in combined:
                    combined[fn]["final_score"] += beta * (res["keyword_score"] / max_keyword)
                else:
                    combined[fn] = {
                        "meta": res["meta"],
                        "document": res["document"],
                        "final_score": beta * (res["keyword_score"] / max_keyword),
                        "keywords": keywords
                    }
            
            for res in bm25_results:
                fn = res["meta"].get("file_name", "unknown")
                if fn in combined:
                    combined[fn]["final_score"] += gamma * (res["bm25_score"] / max_bm25)
                else:
                    combined[fn] = {
                        "meta": res["meta"],
                        "document": res["document"],
                        "final_score": gamma * (res["bm25_score"] / max_bm25),
                        "keywords": keywords
                    }
            
            sorted_results = sorted(combined.values(), key=lambda x: x["final_score"], reverse=True)
            return sorted_results[:self.settings['top_k']]
            
        except Exception as e:
            self.error_occurred.emit(f"Context retrieval error: {e}")
            return []
    
    def rewrite_input_and_generate_synonyms(self):
        try:
            system_prompt = (
                "You are a helpful assistant. Your tasks are:\n"
                "1) Rephrase the given input to make it clearer in one sentence.\n"
                "2) Provide synonyms, spelling variants, plural/singular forms for keywords.\n"
                "Respond in JSON format:\n"
                '{"rephrased": "[sentence]", "keywords": {"[word]": {"synonyms": [], "spelling_variants": [], "plural_singular": [], "parts_of_speech": [], "related_terms": []}}}'
            )
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Rewrite and generate synonyms for: "{self.user_input}"'},
                ],
                model=self.settings['groq_rewrite_model'],
                temperature=0.7,
                stream=False,
                response_format={"type": "json_object"},
            )
            
            response_json = chat_completion.choices[0].message.content.strip()
            response_data = json.loads(response_json)
            
            return response_data.get("rephrased", self.user_input), response_data.get("keywords", {})
            
        except Exception as e:
            return self.user_input, {}


class TTSWorker(QThread):
    """Worker thread for TTS"""
    finished = pyqtSignal()
    
    def __init__(self, text, settings):
        super().__init__()
        self.text = text
        self.settings = settings
        
    def run(self):
        try:
            if not self.text.strip():
                return
            
            tts = gTTS(
                text=self.text,
                lang=self.settings.get('tts_lang', 'en'),
                tld=self.settings.get('tts_tld', 'co.uk')
            )
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            
            audio = AudioSegment.from_file(audio_fp, format="mp3")
            audio = audio.speedup(playback_speed=self.settings.get('tts_speed', 1.4))
            audio = audio + (self.settings.get('tts_volume', 0.5) * 10)
            
            play(audio)
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.finished.emit()


class ChromaDBSearchWorker(QThread):
    """Worker for direct ChromaDB search"""
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
                results["distances"][0]
            ):
                formatted.append({
                    "file_name": meta.get("file_name", "Unknown"),
                    "document": doc,
                    "distance": dist,
                    "similarity": 1.0 - dist,
                    "metadata": meta
                })
            
            self.results_ready.emit(formatted)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ============================================================================
# MAIN WINDOW
# ============================================================================

class RAGChatGUI(QMainWindow):
    def __init__(self):
        print("  __init__ started")
        super().__init__()
        self.setWindowTitle("RAG Chat Assistant")
        self.setGeometry(100, 100, 1400, 900)
        print("  Window geometry set")
        
        # State
        self.collection = None
        self.chromadb_client = None
        self.conversation_history = []
        self.chat_worker = None
        self.tts_worker = None
        self.is_dark_theme = True
        self.tts_enabled = True
        
        # Default settings
        self.settings = {
            'embedding_model': 'mxbai-embed-large',
            'groq_model': 'deepseek-r1-distill-llama-70b',
            'groq_rewrite_model': 'llama-3.3-70b-versatile',
            'ollama_model': 'phi-3',
            'collection_name': 'html_chunks_text_in_documents',
            'chromadb_path': r'C:\Windows_software\easy-local-rag\chroma',
            'system_message': 'You are a helpful assistant. You will give precise and concise answers from the given context. If the context does not have the answer then give it from your knowledge.',
            'top_k': 5,
            'additional_unique_files': 5,
            'alpha': 0.5,
            'beta': 0.3,
            'gamma': 0.2,
            'lambda_mmr': 0.5,
            'tts_speed': 1.4,
            'tts_volume': 0.5,
            'tts_lang': 'en',
            'tts_tld': 'co.uk',
            'just_search': False,
        }
        print("  Settings initialized")
        
        print("  Calling init_ui...")
        self.init_ui()
        print("  Calling apply_dark_theme...")
        self.apply_dark_theme()
        print("  Calling load_settings...")
        self.load_settings()
        print("  __init__ complete - DB will connect after show")
        
    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Settings
        settings_widget = self.create_settings_panel()
        splitter.addWidget(settings_widget)
        
        # Center panel - Chat
        chat_widget = self.create_chat_panel()
        splitter.addWidget(chat_widget)
        
        # Right panel - Context/Search
        context_widget = self.create_context_panel()
        splitter.addWidget(context_widget)
        
        # Set splitter sizes
        splitter.setSizes([300, 600, 400])
        
        # Toolbar
        self.create_toolbar()
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar.addPermanentWidget(self.progress_bar)
        self.statusBar.showMessage("Ready")
        
    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Theme toggle
        theme_action = QAction("🌙 Toggle Theme", self)
        theme_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(theme_action)
        
        # TTS toggle
        self.tts_action = QAction("🔊 TTS On", self)
        self.tts_action.triggered.connect(self.toggle_tts)
        toolbar.addAction(self.tts_action)
        
        toolbar.addSeparator()
        
        # Clear chat
        clear_action = QAction("🗑️ Clear Chat", self)
        clear_action.triggered.connect(self.clear_chat)
        toolbar.addAction(clear_action)
        
        # Export chat
        export_action = QAction("💾 Export Chat", self)
        export_action.triggered.connect(self.export_chat)
        toolbar.addAction(export_action)
        
        toolbar.addSeparator()
        
        # Reconnect DB
        reconnect_action = QAction("🔄 Reconnect DB", self)
        reconnect_action.triggered.connect(self.connect_chromadb)
        toolbar.addAction(reconnect_action)
        
        # Save settings
        save_settings_action = QAction("⚙️ Save Settings", self)
        save_settings_action.triggered.connect(self.save_settings)
        toolbar.addAction(save_settings_action)
        
    def create_settings_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(400)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Models Group
        models_group = QGroupBox("🤖 Models")
        models_layout = QFormLayout()
        
        self.embedding_model_combo = QComboBox()
        self.embedding_model_combo.addItems(['mxbai-embed-large', 'nomic-embed-text', 'all-minilm'])
        self.embedding_model_combo.setEditable(True)
        models_layout.addRow("Embedding:", self.embedding_model_combo)
        
        self.groq_model_combo = QComboBox()
        self.groq_model_combo.addItems([
            'deepseek-r1-distill-llama-70b',
            'llama-3.3-70b-versatile',
            'llama3-70b-8192',
            'mixtral-8x7b-32768'
        ])
        self.groq_model_combo.setEditable(True)
        models_layout.addRow("Groq Model:", self.groq_model_combo)
        
        self.groq_rewrite_combo = QComboBox()
        self.groq_rewrite_combo.addItems([
            'llama-3.3-70b-versatile',
            'llama3-70b-8192',
            'mixtral-8x7b-32768'
        ])
        self.groq_rewrite_combo.setEditable(True)
        models_layout.addRow("Rewrite Model:", self.groq_rewrite_combo)
        
        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.addItems(['phi-3', 'llama3', 'mistral', 'gemma'])
        self.ollama_model_combo.setEditable(True)
        models_layout.addRow("Ollama Model:", self.ollama_model_combo)
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # ChromaDB Group
        db_group = QGroupBox("🗄️ ChromaDB")
        db_layout = QFormLayout()
        
        self.chromadb_path_edit = QLineEdit()
        self.chromadb_path_edit.setText(self.settings['chromadb_path'])
        db_layout.addRow("Path:", self.chromadb_path_edit)
        
        path_btn = QPushButton("Browse...")
        path_btn.clicked.connect(self.browse_chromadb_path)
        db_layout.addRow("", path_btn)
        
        self.collection_combo = QComboBox()
        self.collection_combo.setEditable(True)
        db_layout.addRow("Collection:", self.collection_combo)
        
        refresh_btn = QPushButton("🔄 Refresh Collections")
        refresh_btn.clicked.connect(self.refresh_collections)
        db_layout.addRow("", refresh_btn)
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)
        
        # Search Parameters Group
        search_group = QGroupBox("🔍 Search Parameters")
        search_layout = QFormLayout()
        
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 50)
        self.top_k_spin.setValue(5)
        search_layout.addRow("Top K:", self.top_k_spin)
        
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0, 1)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(0.5)
        search_layout.addRow("Alpha (Vector):", self.alpha_spin)
        
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0, 1)
        self.beta_spin.setSingleStep(0.1)
        self.beta_spin.setValue(0.3)
        search_layout.addRow("Beta (Keyword):", self.beta_spin)
        
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0, 1)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(0.2)
        search_layout.addRow("Gamma (BM25):", self.gamma_spin)
        
        self.lambda_mmr_spin = QDoubleSpinBox()
        self.lambda_mmr_spin.setRange(0, 1)
        self.lambda_mmr_spin.setSingleStep(0.1)
        self.lambda_mmr_spin.setValue(0.5)
        search_layout.addRow("Lambda MMR:", self.lambda_mmr_spin)
        
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
        
        # TTS Group
        tts_group = QGroupBox("🔊 Text-to-Speech")
        tts_layout = QFormLayout()
        
        self.tts_speed_spin = QDoubleSpinBox()
        self.tts_speed_spin.setRange(0.5, 3.0)
        self.tts_speed_spin.setSingleStep(0.1)
        self.tts_speed_spin.setValue(1.4)
        tts_layout.addRow("Speed:", self.tts_speed_spin)
        
        self.tts_volume_spin = QDoubleSpinBox()
        self.tts_volume_spin.setRange(0, 1)
        self.tts_volume_spin.setSingleStep(0.1)
        self.tts_volume_spin.setValue(0.5)
        tts_layout.addRow("Volume:", self.tts_volume_spin)
        
        self.tts_lang_combo = QComboBox()
        self.tts_lang_combo.addItems(['en', 'es', 'fr', 'de', 'it', 'pt'])
        tts_layout.addRow("Language:", self.tts_lang_combo)
        
        self.tts_tld_combo = QComboBox()
        self.tts_tld_combo.addItems(['co.uk', 'com', 'com.au', 'co.in'])
        tts_layout.addRow("Accent:", self.tts_tld_combo)
        
        tts_group.setLayout(tts_layout)
        layout.addWidget(tts_group)
        
        # System Message Group
        sys_group = QGroupBox("📝 System Message")
        sys_layout = QVBoxLayout()
        
        self.system_message_edit = QTextEdit()
        self.system_message_edit.setMaximumHeight(100)
        self.system_message_edit.setText(self.settings['system_message'])
        sys_layout.addWidget(self.system_message_edit)
        
        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)
        
        layout.addStretch()
        scroll.setWidget(widget)
        return scroll
        
    def create_chat_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 11))
        layout.addWidget(self.chat_display, 1)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(80)
        self.chat_input.setPlaceholderText("Type your message here... (Ctrl+Enter to send)")
        input_layout.addWidget(self.chat_input, 1)
        
        btn_layout = QVBoxLayout()
        
        self.send_btn = QPushButton("📤 Send")
        self.send_btn.setMinimumHeight(35)
        self.send_btn.clicked.connect(self.send_message)
        btn_layout.addWidget(self.send_btn)
        
        self.search_only_btn = QPushButton("🔍 Search Only")
        self.search_only_btn.setMinimumHeight(35)
        self.search_only_btn.clicked.connect(self.search_only)
        btn_layout.addWidget(self.search_only_btn)
        
        input_layout.addLayout(btn_layout)
        layout.addLayout(input_layout)
        
        # Quick actions
        quick_layout = QHBoxLayout()
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)
        quick_layout.addWidget(self.stop_btn)
        
        self.read_btn = QPushButton("🔊 Read Last")
        self.read_btn.clicked.connect(self.read_last_response)
        quick_layout.addWidget(self.read_btn)
        
        quick_layout.addStretch()
        layout.addLayout(quick_layout)
        
        return widget
        
    def create_context_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Tabs for different views
        tabs = QTabWidget()
        
        # Context tab
        context_tab = QWidget()
        context_layout = QVBoxLayout(context_tab)
        
        self.context_display = QTextEdit()
        self.context_display.setReadOnly(True)
        self.context_display.setFont(QFont("Consolas", 10))
        context_layout.addWidget(self.context_display)
        
        tabs.addTab(context_tab, "📄 Context")
        
        # Direct Search tab
        search_tab = QWidget()
        search_layout = QVBoxLayout(search_tab)
        
        search_input_layout = QHBoxLayout()
        self.direct_search_input = QLineEdit()
        self.direct_search_input.setPlaceholderText("Direct ChromaDB search...")
        self.direct_search_input.returnPressed.connect(self.direct_chromadb_search)
        search_input_layout.addWidget(self.direct_search_input)
        
        self.direct_search_btn = QPushButton("🔍")
        self.direct_search_btn.clicked.connect(self.direct_chromadb_search)
        search_input_layout.addWidget(self.direct_search_btn)
        
        search_layout.addLayout(search_input_layout)
        
        self.search_results_table = QTableWidget()
        self.search_results_table.setColumnCount(3)
        self.search_results_table.setHorizontalHeaderLabels(["File", "Similarity", "Preview"])
        self.search_results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.search_results_table.itemDoubleClicked.connect(self.show_search_result_detail)
        search_layout.addWidget(self.search_results_table)
        
        tabs.addTab(search_tab, "🔎 Direct Search")
        
        # Collection Browser tab
        browser_tab = QWidget()
        browser_layout = QVBoxLayout(browser_tab)
        
        browser_controls = QHBoxLayout()
        self.browse_count_spin = QSpinBox()
        self.browse_count_spin.setRange(10, 500)
        self.browse_count_spin.setValue(50)
        browser_controls.addWidget(QLabel("Count:"))
        browser_controls.addWidget(self.browse_count_spin)
        
        browse_btn = QPushButton("📂 Browse Collection")
        browse_btn.clicked.connect(self.browse_collection)
        browser_controls.addWidget(browse_btn)
        browser_controls.addStretch()
        
        browser_layout.addLayout(browser_controls)
        
        self.collection_tree = QTreeWidget()
        self.collection_tree.setHeaderLabels(["File Name", "ID", "Info"])
        self.collection_tree.itemDoubleClicked.connect(self.show_collection_item)
        browser_layout.addWidget(self.collection_tree)
        
        tabs.addTab(browser_tab, "📁 Collection Browser")
        
        # Stats tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        stats_layout.addWidget(self.stats_display)
        
        refresh_stats_btn = QPushButton("🔄 Refresh Stats")
        refresh_stats_btn.clicked.connect(self.refresh_stats)
        stats_layout.addWidget(refresh_stats_btn)
        
        tabs.addTab(stats_tab, "📊 Stats")
        
        layout.addWidget(tabs)
        return widget
    
    # =========================================================================
    # SETTINGS MANAGEMENT
    # =========================================================================
    
    def update_settings_from_ui(self):
        self.settings['embedding_model'] = self.embedding_model_combo.currentText()
        self.settings['groq_model'] = self.groq_model_combo.currentText()
        self.settings['groq_rewrite_model'] = self.groq_rewrite_combo.currentText()
        self.settings['ollama_model'] = self.ollama_model_combo.currentText()
        self.settings['chromadb_path'] = self.chromadb_path_edit.text()
        self.settings['collection_name'] = self.collection_combo.currentText()
        self.settings['system_message'] = self.system_message_edit.toPlainText()
        self.settings['top_k'] = self.top_k_spin.value()
        self.settings['alpha'] = self.alpha_spin.value()
        self.settings['beta'] = self.beta_spin.value()
        self.settings['gamma'] = self.gamma_spin.value()
        self.settings['lambda_mmr'] = self.lambda_mmr_spin.value()
        self.settings['tts_speed'] = self.tts_speed_spin.value()
        self.settings['tts_volume'] = self.tts_volume_spin.value()
        self.settings['tts_lang'] = self.tts_lang_combo.currentText()
        self.settings['tts_tld'] = self.tts_tld_combo.currentText()
        
    def save_settings(self):
        self.update_settings_from_ui()
        settings_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv[0] else 'rag_gui.py')), 'rag_gui_settings.json')
        try:
            with open(settings_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            self.statusBar.showMessage("Settings saved!", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save settings: {e}")
            
    def load_settings(self):
        settings_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv[0] else 'rag_gui.py')), 'rag_gui_settings.json')
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    loaded = json.load(f)
                    self.settings.update(loaded)
                    
                # Update UI
                self.embedding_model_combo.setCurrentText(self.settings['embedding_model'])
                self.groq_model_combo.setCurrentText(self.settings['groq_model'])
                self.groq_rewrite_combo.setCurrentText(self.settings['groq_rewrite_model'])
                self.ollama_model_combo.setCurrentText(self.settings['ollama_model'])
                self.chromadb_path_edit.setText(self.settings['chromadb_path'])
                self.system_message_edit.setText(self.settings['system_message'])
                self.top_k_spin.setValue(self.settings['top_k'])
                self.alpha_spin.setValue(self.settings['alpha'])
                self.beta_spin.setValue(self.settings['beta'])
                self.gamma_spin.setValue(self.settings['gamma'])
                self.lambda_mmr_spin.setValue(self.settings['lambda_mmr'])
                self.tts_speed_spin.setValue(self.settings['tts_speed'])
                self.tts_volume_spin.setValue(self.settings['tts_volume'])
                self.tts_lang_combo.setCurrentText(self.settings['tts_lang'])
                self.tts_tld_combo.setCurrentText(self.settings['tts_tld'])
                
                self.statusBar.showMessage("Settings loaded!", 3000)
            except Exception as e:
                print(f"Failed to load settings: {e}")
    
    # =========================================================================
    # CHROMADB
    # =========================================================================
    
    def browse_chromadb_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select ChromaDB Directory")
        if path:
            self.chromadb_path_edit.setText(path)
            
    def connect_chromadb(self, show_errors=True):
        try:
            self.update_settings_from_ui()
            path = self.settings['chromadb_path']
            
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
                
                # Set default if exists
                idx = self.collection_combo.findText(self.settings['collection_name'])
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
    
    # =========================================================================
    # CHAT FUNCTIONALITY
    # =========================================================================
    
    def send_message(self):
        self.settings['just_search'] = False
        self._do_send()
        
    def search_only(self):
        self.settings['just_search'] = True
        self._do_send()
        
    def _do_send(self):
        user_input = self.chat_input.toPlainText().strip()
        if not user_input:
            return
            
        if not self.collection:
            QMessageBox.warning(self, "Error", "Please connect to ChromaDB first!")
            return
            
        self.update_settings_from_ui()
        
        # Display user message
        self.append_message("You", user_input, "#4A90D9")
        self.chat_input.clear()
        
        # Start worker
        self.send_btn.setEnabled(False)
        self.search_only_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        self.chat_worker = ChatWorker(
            user_input,
            self.settings,
            self.collection,
            self.conversation_history
        )
        self.chat_worker.response_chunk.connect(self.on_response_chunk)
        self.chat_worker.response_complete.connect(self.on_response_complete)
        self.chat_worker.context_ready.connect(self.on_context_ready)
        self.chat_worker.error_occurred.connect(self.on_error)
        self.chat_worker.status_update.connect(self.on_status_update)
        self.chat_worker.start()
        
        # Add assistant placeholder
        if not self.settings['just_search']:
            self.append_message("Assistant", "", "#2ECC71", start_only=True)
        
    def on_response_chunk(self, chunk):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(chunk)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()
        
    def on_response_complete(self, response):
        self.send_btn.setEnabled(True)
        self.search_only_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.statusBar.showMessage("Ready", 3000)
        
        if response and self.tts_enabled:
            # Split into sentences for TTS
            sentences = re.split(r'[.!?]+', response)
            for sentence in sentences[:3]:  # Read first 3 sentences
                if sentence.strip():
                    self.tts_worker = TTSWorker(sentence.strip(), self.settings)
                    self.tts_worker.start()
        
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
                    rf'\b({re.escape(kw)})\b',
                    r'<span style="background-color: #FFD700; color: black;">\1</span>',
                    doc,
                    flags=re.IGNORECASE
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
        self.send_btn.setEnabled(True)
        self.search_only_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self.append_message("System", f"Error: {error}", "#E74C3C")
        self.statusBar.showMessage(f"Error: {error}", 5000)
        
    def on_status_update(self, status):
        self.statusBar.showMessage(status)
        
    def stop_generation(self):
        if self.chat_worker and self.chat_worker.isRunning():
            self.chat_worker.terminate()
            self.chat_worker.wait()
            self.on_response_complete("")
            self.append_message("System", "Generation stopped.", "#F39C12")
            
    def append_message(self, sender, message, color, start_only=False):
        timestamp = datetime.now().strftime("%H:%M")
        
        if start_only:
            html = f'<p style="margin: 5px 0;"><span style="color: {color}; font-weight: bold;">[{timestamp}] {sender}:</span> '
        else:
            html = f'<p style="margin: 5px 0;"><span style="color: {color}; font-weight: bold;">[{timestamp}] {sender}:</span> {message}</p>'
        
        self.chat_display.append(html)
        self.chat_display.ensureCursorVisible()
        
    def clear_chat(self):
        reply = QMessageBox.question(
            self, "Clear Chat",
            "Are you sure you want to clear the chat history?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.chat_display.clear()
            self.conversation_history.clear()
            self.context_display.clear()
            self.statusBar.showMessage("Chat cleared", 3000)
            
    def export_chat(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Chat",
            f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Files (*.html);;Text Files (*.txt);;JSON Files (*.json)"
        )
        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(self.conversation_history, f, indent=2)
                elif filename.endswith('.html'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(self.chat_display.toHtml())
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(self.chat_display.toPlainText())
                        
                self.statusBar.showMessage(f"Chat exported to {filename}", 5000)
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Failed to export: {e}")
                
    def read_last_response(self):
        if self.conversation_history:
            for msg in reversed(self.conversation_history):
                if msg['role'] == 'assistant':
                    self.tts_worker = TTSWorker(msg['content'][:500], self.settings)
                    self.tts_worker.start()
                    break
    
    # =========================================================================
    # DIRECT CHROMADB SEARCH
    # =========================================================================
    
    def direct_chromadb_search(self):
        query = self.direct_search_input.text().strip()
        if not query or not self.collection:
            return
            
        self.update_settings_from_ui()
        self.statusBar.showMessage("Searching...")
        
        self.search_worker = ChromaDBSearchWorker(
            query,
            self.collection,
            self.settings['embedding_model'],
            n_results=30
        )
        self.search_worker.results_ready.connect(self.on_search_results)
        self.search_worker.error_occurred.connect(self.on_error)
        self.search_worker.start()
        
    def on_search_results(self, results):
        self.search_results_table.setRowCount(len(results))
        
        for i, res in enumerate(results):
            self.search_results_table.setItem(i, 0, QTableWidgetItem(res['file_name']))
            self.search_results_table.setItem(i, 1, QTableWidgetItem(f"{res['similarity']:.4f}"))
            self.search_results_table.setItem(i, 2, QTableWidgetItem(res['document'][:100]))
            
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
    
    # =========================================================================
    # COLLECTION BROWSER
    # =========================================================================
    
    def browse_collection(self):
        if not self.collection:
            return
            
        self.collection_tree.clear()
        count = self.browse_count_spin.value()
        
        try:
            results = self.collection.get(
                limit=count,
                include=["metadatas", "documents"]
            )
            
            # Group by file
            files = {}
            for id_, meta, doc in zip(results['ids'], results['metadatas'], results['documents']):
                fn = meta.get('file_name', 'Unknown')
                if fn not in files:
                    files[fn] = []
                files[fn].append({'id': id_, 'meta': meta, 'doc': doc})
            
            for fn, items in files.items():
                parent = QTreeWidgetItem([fn, "", f"{len(items)} chunks"])
                self.collection_tree.addTopLevelItem(parent)
                
                for item in items:
                    child = QTreeWidgetItem([
                        item['doc'][:50] + "...",
                        item['id'],
                        ""
                    ])
                    child.setData(0, Qt.UserRole, item)
                    parent.addChild(child)
                    
            self.statusBar.showMessage(f"Loaded {len(results['ids'])} items from {len(files)} files", 3000)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to browse collection: {e}")
            
    def show_collection_item(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data:
            detail = f"""
ID: {data['id']}

--- Document Content ---
{data['doc']}

--- Metadata ---
{json.dumps(data['meta'], indent=2, default=str)}
"""
            QMessageBox.information(self, "Document Details", detail)
    
    # =========================================================================
    # THEME & UI
    # =========================================================================
    
    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        if self.is_dark_theme:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()
            
    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1E1E1E;
                color: #D4D4D4;
            }
            QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #2D2D2D;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
                padding: 4px;
            }
            QTextEdit:focus, QLineEdit:focus {
                border: 1px solid #007ACC;
            }
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1177BB;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
            QPushButton:disabled {
                background-color: #3C3C3C;
                color: #808080;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3C3C3C;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #3C3C3C;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: #D4D4D4;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #1E1E1E;
                border-bottom: 2px solid #007ACC;
            }
            QScrollBar:vertical {
                background-color: #2D2D2D;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #3C3C3C;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4C4C4C;
            }
            QTableWidget {
                background-color: #2D2D2D;
                gridline-color: #3C3C3C;
            }
            QHeaderView::section {
                background-color: #252526;
                color: #D4D4D4;
                padding: 5px;
                border: 1px solid #3C3C3C;
            }
            QTreeWidget {
                background-color: #2D2D2D;
            }
            QTreeWidget::item:hover {
                background-color: #3C3C3C;
            }
            QTreeWidget::item:selected {
                background-color: #094771;
            }
            QToolBar {
                background-color: #252526;
                border: none;
                spacing: 5px;
            }
            QStatusBar {
                background-color: #007ACC;
                color: white;
            }
            QProgressBar {
                border: none;
                background-color: #3C3C3C;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #0E639C;
                border-radius: 4px;
            }
        """)
        
    def apply_light_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #F5F5F5;
                color: #333333;
            }
            QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: white;
                color: #333333;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 4px;
            }
            QTextEdit:focus, QLineEdit:focus {
                border: 1px solid #0078D4;
            }
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #808080;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #E5E5E5;
                color: #333333;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #F5F5F5;
                border-bottom: 2px solid #0078D4;
            }
            QTableWidget {
                background-color: white;
                gridline-color: #CCCCCC;
            }
            QHeaderView::section {
                background-color: #E5E5E5;
                color: #333333;
                padding: 5px;
                border: 1px solid #CCCCCC;
            }
            QTreeWidget {
                background-color: white;
            }
            QTreeWidget::item:hover {
                background-color: #E5E5E5;
            }
            QTreeWidget::item:selected {
                background-color: #CCE4F7;
            }
            QToolBar {
                background-color: #E5E5E5;
                border: none;
                spacing: 5px;
            }
            QStatusBar {
                background-color: #0078D4;
                color: white;
            }
        """)
        
    def toggle_tts(self):
        self.tts_enabled = not self.tts_enabled
        if self.tts_enabled:
            self.tts_action.setText("🔊 TTS On")
        else:
            self.tts_action.setText("🔇 TTS Off")
            
    def keyPressEvent(self, event):
        # Ctrl+Enter to send
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
            self.send_message()
        else:
            super().keyPressEvent(event)


# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        print("Starting app...")
        app = QApplication(sys.argv)
        print("QApplication created")
        app.setStyle('Fusion')
        print("Creating window...")
        window = RAGChatGUI()
        print("Window created, showing...")
        window.show()
        print("Window shown, connecting to DB...")
        # Connect to DB after window is shown
        QTimer.singleShot(100, window.connect_chromadb)
        print("Entering event loop...")
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        print("=" * 50)
        print("ERROR OCCURRED:")
        print("=" * 50)
        traceback.print_exc()
        print("=" * 50)
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
