
import sys
import os
import json
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
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QTextCharFormat

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

import ollama
from groq import Groq

import numpy as np
from rank_bm25 import BM25Okapi
import subprocess

from GUI_workers import TTSWorker, ChromaDBSearchWorker
from GUI_chat import ChatFunctionalityMixin

import GUI_settings
import GUI_theme
import GUI_context

import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()


# =========================================================================
# GLOBAL EXCEPTION HOOK FOR DEBUGGING
# =========================================================================

def _global_excepthook(exc_type, exc_value, exc_traceback):
    """Log any uncaught exceptions to help debug crashes."""
    import traceback as _tb
    print("\n========== UNCAUGHT EXCEPTION ==========")
    _tb.print_exception(exc_type, exc_value, exc_traceback)
    print("=======================================\n")


sys.excepthook = _global_excepthook


 # ============================================================================
 # WORKER THREADS
 # ============================================================================


# ============================================================================
# MAIN WINDOW
# ============================================================================

class RAGChatGUI(ChatFunctionalityMixin, QMainWindow):
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
        self.tts_workers = []  # Track all TTS workers
        self.is_dark_theme = True
        self.tts_enabled = True
        
        # Default settings
        self.settings = GUI_settings.DEFAULT_SETTINGS.copy()
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
        self.groq_model_combo.setEditable(True)
        models_layout.addRow("Groq Model:", self.groq_model_combo)
        
        self.groq_rewrite_combo = QComboBox()
        self.groq_rewrite_combo.setEditable(True)
        models_layout.addRow("Rewrite Model:", self.groq_rewrite_combo)
        
        # Refresh models button
        refresh_models_btn = QPushButton("Refresh Groq Models")
        refresh_models_btn.clicked.connect(self.load_groq_models)
        models_layout.addRow("", refresh_models_btn)
        
        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.setEditable(True)
        models_layout.addRow("Ollama Model:", self.ollama_model_combo)
        
        # Refresh Ollama models button
        refresh_ollama_btn = QPushButton("Refresh Ollama Models")
        refresh_ollama_btn.clicked.connect(self.load_ollama_models)
        models_layout.addRow("", refresh_ollama_btn)
        
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
        GUI_settings.update_settings_from_ui(self, self.settings)
        
    def save_settings(self):
        GUI_settings.save_settings(self, self.settings)
            
    def load_settings(self):
        GUI_settings.load_settings(self, self.settings)
    
    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    
    def load_groq_models(self):
        """Dynamically load available models from Groq API"""
        try:
            self.statusBar.showMessage("Loading Groq models...")
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            models_response = groq_client.models.list()
            
            # Filter for chat models (exclude whisper, tts, etc.)
            chat_models = []
            for model in models_response.data:
                model_id = model.id
                # Skip audio/speech models
                if any(x in model_id.lower() for x in ['whisper', 'tts', 'guard', 'safeguard']):
                    continue
                chat_models.append(model_id)
            
            # Sort models
            chat_models.sort()
            
            # Save current selections
            current_groq = self.groq_model_combo.currentText()
            current_rewrite = self.groq_rewrite_combo.currentText()
            
            # Update combo boxes
            self.groq_model_combo.clear()
            self.groq_rewrite_combo.clear()
            
            self.groq_model_combo.addItems(chat_models)
            self.groq_rewrite_combo.addItems(chat_models)
            
            # Restore selections if they exist
            idx = self.groq_model_combo.findText(current_groq)
            if idx >= 0:
                self.groq_model_combo.setCurrentIndex(idx)
            else:
                # Set default to llama-3.3-70b-versatile if available
                idx = self.groq_model_combo.findText('llama-3.3-70b-versatile')
                if idx >= 0:
                    self.groq_model_combo.setCurrentIndex(idx)
            
            idx = self.groq_rewrite_combo.findText(current_rewrite)
            if idx >= 0:
                self.groq_rewrite_combo.setCurrentIndex(idx)
            else:
                idx = self.groq_rewrite_combo.findText('llama-3.1-8b-instant')
                if idx >= 0:
                    self.groq_rewrite_combo.setCurrentIndex(idx)
            
            self.statusBar.showMessage(f"Loaded {len(chat_models)} Groq models", 3000)
            
        except Exception as e:
            self.statusBar.showMessage(f"Failed to load Groq models: {e}", 5000)
            # Fallback to default models
            default_models = [
                'llama-3.3-70b-versatile',
                'llama-3.1-8b-instant',
                'openai/gpt-oss-120b',
                'openai/gpt-oss-20b',
                'qwen/qwen3-32b',
                'meta-llama/llama-4-maverick-17b-128e-instruct',
                'meta-llama/llama-4-scout-17b-16e-instruct',
            ]
            if self.groq_model_combo.count() == 0:
                self.groq_model_combo.addItems(default_models)
                self.groq_rewrite_combo.addItems(default_models)
    
    def load_ollama_models(self):
        """Dynamically load available models from Ollama"""
        try:
            self.statusBar.showMessage("Loading Ollama models...")
            models_list = ollama.list()
            
            model_names = [m['name'] for m in models_list.get('models', [])]
            model_names.sort()
            
            current = self.ollama_model_combo.currentText()
            self.ollama_model_combo.clear()
            
            if model_names:
                self.ollama_model_combo.addItems(model_names)
                idx = self.ollama_model_combo.findText(current)
                if idx >= 0:
                    self.ollama_model_combo.setCurrentIndex(idx)
                self.statusBar.showMessage(f"Loaded {len(model_names)} Ollama models", 3000)
            else:
                self.ollama_model_combo.addItems(['phi-3', 'llama3', 'mistral', 'gemma'])
                self.statusBar.showMessage("No Ollama models found, using defaults", 3000)
                
        except Exception as e:
            self.statusBar.showMessage(f"Failed to load Ollama models: {e}", 5000)
    
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
    # CONTEXT RETRIEVAL (Main thread - ChromaDB not thread-safe)
    # =========================================================================
    
    def get_relevant_context_hybrid(self, user_input):
        """Get relevant context using hybrid search - runs in main thread"""
        return GUI_context.get_relevant_context_hybrid(self.settings, user_input)
    
    def rewrite_input_and_generate_synonyms(self, user_input):
        """Rewrite query and generate synonyms using Groq"""
        return GUI_context.rewrite_input_and_generate_synonyms(self.settings, user_input)
    
    def closeEvent(self, event):
        """Properly clean up threads before closing"""
        print("[closeEvent] Waiting for threads to finish...")
        
        # Stop and wait for chat worker
        if self.chat_worker and self.chat_worker.isRunning():
            print("[closeEvent] Terminating ChatWorker")
            self.chat_worker.terminate()
            self.chat_worker.wait(2000)
        
        # Wait for TTS workers
        for worker in self.tts_workers[:]:  # Copy list to avoid modification during iteration
            if worker.isRunning():
                print("[closeEvent] Waiting for TTS worker")
                worker.wait(1000)
            worker.deleteLater()
        self.tts_workers.clear()
        
        print("[closeEvent] Cleanup done, closing window")
        event.accept()
        
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
        GUI_theme.apply_dark_theme(self)
        
    def apply_light_theme(self):
        GUI_theme.apply_light_theme(self)
        
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
        print("Window shown, loading models and connecting to DB...")
        # Load models and connect to DB after window is shown
        QTimer.singleShot(100, window.load_groq_models)
        QTimer.singleShot(200, window.load_ollama_models)
        QTimer.singleShot(300, window.connect_chromadb)
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
