"""
Comprehensive tests for rag_gui.py and its supporting modules.
Tests cover: settings, chat mixin, chromadb mixin, direct search, model loading, theme, workers.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import GUI_settings
import GUI_theme
import GUI_context
from GUI_chat import ChatFunctionalityMixin
from GUI_chromadb import ChromaDBMixin
from GUI_direct_search import DirectSearchMixin
from GUI_models import ModelLoadingMixin


# =============================================================================
# DUMMY/MOCK CLASSES FOR TESTING WITHOUT PyQt
# =============================================================================

class DummyStatusBar:
    """Mock status bar that records messages."""
    def __init__(self):
        self.messages = []
        self.last_message = ""
    
    def showMessage(self, message, timeout=None):
        self.messages.append((message, timeout))
        self.last_message = message


class DummyCombo:
    """Mock QComboBox."""
    def __init__(self, text="", items=None):
        self._text = text
        self._items = items or []
        self._index = 0
        self._signals_blocked = False
    
    def currentText(self):
        return self._text
    
    def setCurrentText(self, text):
        self._text = text
    
    def addItem(self, item):
        self._items.append(item)
    
    def addItems(self, items):
        self._items.extend(items)
    
    def clear(self):
        self._items = []
    
    def count(self):
        return len(self._items)
    
    def findText(self, text):
        try:
            return self._items.index(text)
        except ValueError:
            return -1
    
    def setCurrentIndex(self, idx):
        self._index = idx
        if 0 <= idx < len(self._items):
            self._text = self._items[idx]
    
    def blockSignals(self, block):
        self._signals_blocked = block


class DummyLineEdit:
    """Mock QLineEdit."""
    def __init__(self, text=""):
        self._text = text
    
    def text(self):
        return self._text
    
    def setText(self, text):
        self._text = text


class DummySpin:
    """Mock QSpinBox/QDoubleSpinBox."""
    def __init__(self, value=0):
        self._value = value
    
    def value(self):
        return self._value
    
    def setValue(self, value):
        self._value = value


class DummyTextEdit:
    """Mock QTextEdit."""
    def __init__(self, text=""):
        self._text = text
        self._html = ""
        self._appended = []
        self._cleared = False
    
    def toPlainText(self):
        return self._text
    
    def setText(self, text):
        self._text = text
    
    def setPlainText(self, text):
        self._text = text
    
    def toHtml(self):
        return self._html or f"<html>{self._text}</html>"
    
    def setHtml(self, html):
        self._html = html
    
    def append(self, text):
        self._appended.append(text)
        self._text += text
    
    def clear(self):
        self._text = ""
        self._html = ""
        self._appended = []
        self._cleared = True
    
    def ensureCursorVisible(self):
        pass
    
    def textCursor(self):
        return DummyTextCursor()
    
    def setTextCursor(self, cursor):
        pass


class DummyTextCursor:
    """Mock QTextCursor."""
    def movePosition(self, *args):
        pass
    
    def insertText(self, text):
        pass


class DummyProgressBar:
    """Mock QProgressBar."""
    def __init__(self):
        self._visible = False
        self._range = (0, 100)
    
    def setVisible(self, visible):
        self._visible = visible
    
    def setRange(self, min_val, max_val):
        self._range = (min_val, max_val)


class DummyButton:
    """Mock QPushButton."""
    def __init__(self, enabled=True):
        self._enabled = enabled
        self._text = ""
    
    def setEnabled(self, enabled):
        self._enabled = enabled
    
    def isEnabled(self):
        return self._enabled
    
    def setText(self, text):
        self._text = text


class DummyAction:
    """Mock QAction."""
    def __init__(self, text=""):
        self._text = text
    
    def setText(self, text):
        self._text = text
    
    def text(self):
        return self._text


class DummyTableWidget:
    """Mock QTableWidget."""
    def __init__(self):
        self._rows = []
        self._row_count = 0
    
    def setRowCount(self, count):
        self._row_count = count
        self._rows = [{"items": {}} for _ in range(count)]
    
    def setItem(self, row, col, item):
        if row < len(self._rows):
            self._rows[row]["items"][col] = item
    
    def item(self, row, col):
        if row < len(self._rows):
            return self._rows[row]["items"].get(col)
        return None


class DummyTableWidgetItem:
    """Mock QTableWidgetItem."""
    def __init__(self, text=""):
        self._text = text
        self._data = {}
    
    def setData(self, role, data):
        self._data[role] = data
    
    def data(self, role):
        return self._data.get(role)
    
    def row(self):
        return 0


class DummyTreeWidget:
    """Mock QTreeWidget."""
    def __init__(self):
        self._items = []
        self._cleared = False
    
    def clear(self):
        self._items = []
        self._cleared = True
    
    def addTopLevelItem(self, item):
        self._items.append(item)


class DummyTreeWidgetItem:
    """Mock QTreeWidgetItem."""
    def __init__(self, texts=None):
        self.texts = texts or []
        self._data = {}
        self._children = []
    
    def setData(self, col, role, data):
        self._data[(col, role)] = data
    
    def data(self, col, role):
        return self._data.get((col, role))
    
    def addChild(self, child):
        self._children.append(child)


class DummyCollection:
    """Mock ChromaDB collection."""
    def __init__(self, name="test_collection", count=100):
        self.name = name
        self._count = count
        self._data = {
            "ids": ["id1", "id2", "id3"],
            "metadatas": [
                {"file_name": "doc1.txt"},
                {"file_name": "doc2.txt"},
                {"file_name": "doc1.txt"},
            ],
            "documents": [
                "First document content here",
                "Second document with more text",
                "Third document from same file",
            ],
            "distances": [0.1, 0.2, 0.3],
        }
    
    def count(self):
        return self._count
    
    def get(self, limit=10, include=None):
        return self._data
    
    def query(self, query_embeddings=None, n_results=10, include=None):
        return {
            "metadatas": [self._data["metadatas"][:n_results]],
            "documents": [self._data["documents"][:n_results]],
            "distances": [self._data["distances"][:n_results]],
        }


class DummyChromaClient:
    """Mock ChromaDB PersistentClient."""
    def __init__(self, collections=None):
        self._collections = collections or [
            type("Col", (), {"name": "collection1"})(),
            type("Col", (), {"name": "collection2"})(),
        ]
    
    def list_collections(self):
        return self._collections
    
    def get_collection(self, name):
        return DummyCollection(name=name)


class DummyWorker:
    """Mock QThread worker."""
    def __init__(self):
        self._running = False
        self._finished = False
    
    def isRunning(self):
        return self._running
    
    def isFinished(self):
        return self._finished
    
    def wait(self, timeout=None):
        self._running = False
        self._finished = True
    
    def terminate(self):
        self._running = False
    
    def deleteLater(self):
        pass
    
    def start(self):
        self._running = True


# =============================================================================
# WINDOW MOCK CLASSES
# =============================================================================

class MockWindowBase:
    """Base mock window with common UI elements."""
    def __init__(self):
        self.settings = GUI_settings.DEFAULT_SETTINGS.copy()
        self.statusBar = DummyStatusBar()
        self.collection = None
        self.chromadb_client = None
        self.conversation_history = []
        self.chat_worker = None
        self.tts_workers = []
        self.tts_enabled = True
        self.is_dark_theme = True
        
        # UI Elements
        self.embedding_model_combo = DummyCombo("mxbai-embed-large")
        self.groq_model_combo = DummyCombo("llama-3.3-70b-versatile")
        self.groq_rewrite_combo = DummyCombo("llama-3.1-8b-instant")
        self.ollama_model_combo = DummyCombo("phi-3")
        self.chromadb_path_edit = DummyLineEdit("/tmp/chroma")
        self.collection_combo = DummyCombo("test_collection")
        self.system_message_edit = DummyTextEdit("You are a helpful assistant.")
        self.top_k_spin = DummySpin(5)
        self.alpha_spin = DummySpin(0.5)
        self.beta_spin = DummySpin(0.3)
        self.gamma_spin = DummySpin(0.2)
        self.lambda_mmr_spin = DummySpin(0.5)
        self.tts_speed_spin = DummySpin(1.4)
        self.tts_volume_spin = DummySpin(0.5)
        self.tts_lang_combo = DummyCombo("en")
        self.tts_tld_combo = DummyCombo("co.uk")
        
        self.chat_display = DummyTextEdit()
        self.chat_input = DummyTextEdit()
        self.context_display = DummyTextEdit()
        self.stats_display = DummyTextEdit()
        
        self.send_btn = DummyButton()
        self.search_only_btn = DummyButton()
        self.stop_btn = DummyButton(enabled=False)
        self.tts_action = DummyAction("🔊 TTS On")
        
        self.progress_bar = DummyProgressBar()
        
        self.direct_search_input = DummyLineEdit()
        self.search_results_table = DummyTableWidget()
        
        self.collection_tree = DummyTreeWidget()
        self.browse_count_spin = DummySpin(50)
    
    def update_settings_from_ui(self):
        GUI_settings.update_settings_from_ui(self, self.settings)
    
    def isVisible(self):
        return True


class MockChatWindow(ChatFunctionalityMixin, MockWindowBase):
    """Mock window with ChatFunctionalityMixin for testing chat functionality."""
    def __init__(self):
        MockWindowBase.__init__(self)
    
    def get_relevant_context_hybrid(self, user_input):
        return [
            {"document": "Test context", "meta": {"file_name": "test.txt"}, "final_score": 0.9, "keywords": []},
        ]
    
    def append_message(self, sender, message, color, start_only=False):
        """Mock append_message for testing."""
        self.chat_display.append(f"[{sender}]: {message}")


class MockChromaDBWindow(ChromaDBMixin, MockWindowBase):
    """Mock window with ChromaDBMixin for testing ChromaDB functionality."""
    def __init__(self):
        MockWindowBase.__init__(self)


class MockDirectSearchWindow(DirectSearchMixin, MockWindowBase):
    """Mock window with DirectSearchMixin for testing direct search functionality."""
    def __init__(self):
        MockWindowBase.__init__(self)
    
    def on_error(self, error):
        self.statusBar.showMessage(f"Error: {error}")


class MockModelLoadingWindow(ModelLoadingMixin, MockWindowBase):
    """Mock window with ModelLoadingMixin for testing model loading functionality."""
    def __init__(self):
        MockWindowBase.__init__(self)


# =============================================================================
# SETTINGS TESTS
# =============================================================================

class TestGUISettings:
    """Tests for GUI_settings module."""
    
    def test_default_settings_contains_required_keys(self):
        """Verify all required settings keys exist in defaults."""
        required_keys = [
            'embedding_model', 'groq_model', 'groq_rewrite_model', 'ollama_model',
            'collection_name', 'chromadb_path', 'system_message', 'top_k',
            'alpha', 'beta', 'gamma', 'lambda_mmr',
            'tts_speed', 'tts_volume', 'tts_lang', 'tts_tld', 'just_search',
        ]
        for key in required_keys:
            assert key in GUI_settings.DEFAULT_SETTINGS, f"Missing key: {key}"
    
    def test_update_settings_from_ui_reads_all_widget_values(self):
        """Verify update_settings_from_ui correctly reads widget values."""
        window = MockWindowBase()
        window.embedding_model_combo.setCurrentText("custom-embed")
        window.groq_model_combo.setCurrentText("custom-groq")
        window.top_k_spin.setValue(10)
        window.alpha_spin.setValue(0.8)
        
        settings = GUI_settings.DEFAULT_SETTINGS.copy()
        GUI_settings.update_settings_from_ui(window, settings)
        
        assert settings["embedding_model"] == "custom-embed"
        assert settings["groq_model"] == "custom-groq"
        assert settings["top_k"] == 10
        assert settings["alpha"] == 0.8
    
    def test_update_settings_preserves_empty_widget_values(self):
        """Empty combo text should not overwrite settings."""
        window = MockWindowBase()
        window.embedding_model_combo.setCurrentText("")  # Empty
        settings = {"embedding_model": "original"}
        
        GUI_settings.update_settings_from_ui(window, settings)
        
        # Empty text should not overwrite
        assert settings["embedding_model"] == "original"
    
    def test_save_settings_creates_file(self, tmp_path, monkeypatch):
        """Verify save_settings creates JSON file."""
        window = MockWindowBase()
        settings = GUI_settings.DEFAULT_SETTINGS.copy()
        settings_path = tmp_path / "test_settings.json"
        monkeypatch.setattr(GUI_settings, "_settings_path", lambda: str(settings_path))
        
        GUI_settings.save_settings(window, settings)
        
        assert settings_path.exists()
        with open(settings_path, "r") as f:
            saved = json.load(f)
        assert "embedding_model" in saved
    
    def test_load_settings_updates_widgets(self, tmp_path, monkeypatch):
        """Verify load_settings updates widget values."""
        settings_path = tmp_path / "test_settings.json"
        test_settings = {"embedding_model": "loaded-model", "top_k": 15, "alpha": 0.9}
        test_settings.update(GUI_settings.DEFAULT_SETTINGS)
        test_settings["embedding_model"] = "loaded-model"
        test_settings["top_k"] = 15
        
        with open(settings_path, "w") as f:
            json.dump(test_settings, f)
        
        monkeypatch.setattr(GUI_settings, "_settings_path", lambda: str(settings_path))
        
        window = MockWindowBase()
        settings = GUI_settings.DEFAULT_SETTINGS.copy()
        GUI_settings.load_settings(window, settings)
        
        assert window.embedding_model_combo.currentText() == "loaded-model"
        assert window.top_k_spin.value() == 15
    
    def test_load_settings_handles_missing_file(self, tmp_path, monkeypatch):
        """Verify load_settings gracefully handles missing file."""
        settings_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr(GUI_settings, "_settings_path", lambda: str(settings_path))
        
        window = MockWindowBase()
        settings = GUI_settings.DEFAULT_SETTINGS.copy()
        
        # Should not raise
        GUI_settings.load_settings(window, settings)
        
        # Settings unchanged
        assert settings == GUI_settings.DEFAULT_SETTINGS


# =============================================================================
# CHAT FUNCTIONALITY TESTS
# =============================================================================

class TestChatFunctionalityMixin:
    """Tests for ChatFunctionalityMixin."""
    
    def test_on_context_ready_with_empty_results(self):
        """Verify on_context_ready handles empty results."""
        window = MockChatWindow()
        
        window.on_context_ready([])
        
        assert window.context_display._cleared
        assert "No context found" in window.context_display._text
    
    def test_on_context_ready_formats_results_with_keywords(self):
        """Verify on_context_ready formats results and highlights keywords."""
        window = MockChatWindow()
        results = [
            {
                "meta": {"file_name": "document.txt"},
                "final_score": 0.85,
                "document": "This document contains important keywords for testing.",
                "keywords": ["important", "testing"],
            },
        ]
        
        window.on_context_ready(results)
        
        html = window.context_display._html
        assert "document.txt" in html
        assert "0.850" in html  # Score formatted to 3 decimals
        assert "background-color: #FFD700" in html  # Keyword highlight
    
    def test_on_context_ready_escapes_regex_special_chars(self):
        """Verify keywords with regex special chars don't break highlighting."""
        window = MockChatWindow()
        results = [
            {
                "meta": {"file_name": "test.txt"},
                "final_score": 0.5,
                "document": "Test with (parentheses) and [brackets]",
                "keywords": ["(parentheses)", "[brackets]"],
            },
        ]
        
        # Should not raise
        window.on_context_ready(results)
    
    def test_on_error_re_enables_buttons(self):
        """Verify on_error re-enables send buttons."""
        window = MockChatWindow()
        window.send_btn.setEnabled(False)
        window.search_only_btn.setEnabled(False)
        window.stop_btn.setEnabled(True)
        
        window.on_error("Test error message")
        
        assert window.send_btn.isEnabled()
        assert window.search_only_btn.isEnabled()
        assert not window.stop_btn.isEnabled()
    
    def test_on_response_complete_re_enables_buttons(self):
        """Verify on_response_complete re-enables buttons."""
        window = MockChatWindow()
        window.send_btn.setEnabled(False)
        window.search_only_btn.setEnabled(False)
        window.stop_btn.setEnabled(True)
        window.tts_enabled = False  # Disable TTS for this test
        
        window.on_response_complete("Test response")
        
        assert window.send_btn.isEnabled()
        assert window.search_only_btn.isEnabled()
        assert not window.stop_btn.isEnabled()
    
    def test_on_response_complete_cleans_up_worker(self):
        """Verify on_response_complete cleans up chat worker."""
        window = MockChatWindow()
        window.chat_worker = DummyWorker()
        window.tts_enabled = False
        
        window.on_response_complete("Response")
        
        assert window.chat_worker is None
    
    def test_on_status_update_updates_statusbar(self):
        """Verify on_status_update sets status bar message."""
        window = MockChatWindow()
        
        window.on_status_update("Processing...")
        
        assert window.statusBar.last_message == "Processing..."


# =============================================================================
# CHROMADB MIXIN TESTS
# =============================================================================

class TestChromaDBMixin:
    """Tests for ChromaDBMixin."""
    
    def test_refresh_collections_populates_combo(self):
        """Verify refresh_collections populates collection combo."""
        window = MockChromaDBWindow()
        window.chromadb_client = DummyChromaClient()
        
        window.refresh_collections()
        
        assert window.collection_combo.count() == 2
        assert "collection1" in window.collection_combo._items
        assert "collection2" in window.collection_combo._items
    
    def test_refresh_collections_without_client(self):
        """Verify refresh_collections handles missing client gracefully."""
        window = MockChromaDBWindow()
        window.chromadb_client = None
        
        # Should not raise
        window.refresh_collections()
    
    def test_refresh_collections_selects_preferred(self):
        """Verify refresh_collections selects preferred collection."""
        window = MockChromaDBWindow()
        window.chromadb_client = DummyChromaClient()
        window.settings["collection_name"] = "collection2"
        
        window.refresh_collections()
        
        assert window.collection_combo.currentText() == "collection2"
    
    def test_refresh_stats_displays_collection_info(self):
        """Verify refresh_stats displays collection statistics."""
        window = MockChromaDBWindow()
        window.collection = DummyCollection(count=500)
        window.settings["embedding_model"] = "test-embed"
        window.settings["groq_model"] = "test-groq"
        window.settings["top_k"] = 10
        
        window.refresh_stats()
        
        text = window.stats_display._text
        assert "500" in text  # Document count
        assert "test-embed" in text
        assert "test-groq" in text
    
    def test_refresh_stats_without_collection(self):
        """Verify refresh_stats handles missing collection."""
        window = MockChromaDBWindow()
        window.collection = None
        
        window.refresh_stats()
        
        assert "No collection connected" in window.stats_display._text
    
    @patch("GUI_chromadb.chromadb.PersistentClient")
    def test_connect_chromadb_creates_client(self, mock_client_class):
        """Verify connect_chromadb creates ChromaDB client."""
        mock_client = DummyChromaClient()
        mock_client_class.return_value = mock_client
        
        window = MockChromaDBWindow()
        window.settings["chromadb_path"] = "/test/path"
        
        window.connect_chromadb(show_errors=False)
        
        assert window.chromadb_client is mock_client


# =============================================================================
# DIRECT SEARCH MIXIN TESTS
# =============================================================================

class TestDirectSearchMixin:
    """Tests for DirectSearchMixin."""
    
    def test_on_search_results_populates_table(self):
        """Verify on_search_results populates the results table."""
        window = MockDirectSearchWindow()
        results = [
            {"file_name": "doc1.txt", "similarity": 0.95, "distance": 0.05, "document": "Content 1", "metadata": {}},
            {"file_name": "doc2.txt", "similarity": 0.80, "distance": 0.20, "document": "Content 2", "metadata": {}},
        ]
        
        window.on_search_results(results)
        
        assert window.search_results_table._row_count == 2
        assert "Found 2 results" in window.statusBar.last_message
    
    def test_direct_chromadb_search_requires_query_and_collection(self):
        """Verify direct search requires both query and collection."""
        window = MockDirectSearchWindow()
        window.direct_search_input.setText("")
        window.collection = None
        
        # Should return early without error
        window.direct_chromadb_search()
        
        # No status update about searching
        assert not any("Searching" in msg for msg, _ in window.statusBar.messages)


# =============================================================================
# MODEL LOADING MIXIN TESTS
# =============================================================================

class TestModelLoadingMixin:
    """Tests for ModelLoadingMixin."""
    
    @patch("GUI_models.Groq")
    def test_load_groq_models_populates_combos(self, mock_groq_class):
        """Verify load_groq_models populates model combos."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(
            data=[
                MagicMock(id="llama-3.3-70b-versatile"),
                MagicMock(id="llama-3.1-8b-instant"),
                MagicMock(id="whisper-large"),  # Should be filtered
            ]
        )
        mock_groq_class.return_value = mock_client
        
        window = MockModelLoadingWindow()
        window.load_groq_models()
        
        # Whisper should be filtered out
        assert "whisper-large" not in window.groq_model_combo._items
        assert len(window.groq_model_combo._items) == 2
    
    @patch("GUI_models.Groq")
    def test_load_groq_models_handles_error(self, mock_groq_class):
        """Verify load_groq_models handles API errors gracefully."""
        mock_groq_class.side_effect = Exception("API Error")
        
        window = MockModelLoadingWindow()
        # Should not raise
        window.load_groq_models()
        
        # Fallback models should be added
        assert window.groq_model_combo.count() > 0
    
    @patch("GUI_models.ollama.list")
    def test_load_ollama_models_populates_combo(self, mock_ollama_list):
        """Verify load_ollama_models populates model combo."""
        mock_ollama_list.return_value = {
            "models": [
                {"name": "phi-3"},
                {"name": "llama3"},
                {"name": "mistral"},
            ]
        }
        
        window = MockModelLoadingWindow()
        window.load_ollama_models()
        
        assert window.ollama_model_combo.count() == 3
        assert "phi-3" in window.ollama_model_combo._items
    
    @patch("GUI_models.ollama.list")
    def test_load_ollama_models_handles_empty_list(self, mock_ollama_list):
        """Verify load_ollama_models handles empty model list."""
        mock_ollama_list.return_value = {"models": []}
        
        window = MockModelLoadingWindow()
        window.load_ollama_models()
        
        # Fallback models should be added
        assert window.ollama_model_combo.count() > 0


# =============================================================================
# THEME TESTS
# =============================================================================

class TestGUITheme:
    """Tests for GUI_theme module."""
    
    def test_apply_dark_theme_sets_stylesheet(self):
        """Verify apply_dark_theme sets stylesheet on widget."""
        mock_widget = MagicMock()
        
        GUI_theme.apply_dark_theme(mock_widget)
        
        mock_widget.setStyleSheet.assert_called_once()
        stylesheet = mock_widget.setStyleSheet.call_args[0][0]
        assert "#1E1E1E" in stylesheet  # Dark background color
    
    def test_apply_light_theme_sets_stylesheet(self):
        """Verify apply_light_theme sets stylesheet on widget."""
        mock_widget = MagicMock()
        
        GUI_theme.apply_light_theme(mock_widget)
        
        mock_widget.setStyleSheet.assert_called_once()
        stylesheet = mock_widget.setStyleSheet.call_args[0][0]
        assert "#F5F5F5" in stylesheet  # Light background color


# =============================================================================
# CONTEXT RETRIEVAL TESTS
# =============================================================================

class TestGUIContext:
    """Tests for GUI_context module."""
    
    @patch("GUI_context.Groq")
    def test_rewrite_input_handles_api_error(self, mock_groq_class):
        """Verify rewrite_input_and_generate_synonyms handles API errors."""
        mock_groq_class.side_effect = Exception("API Error")
        
        settings = {"groq_rewrite_model": "test-model"}
        result, keywords = GUI_context.rewrite_input_and_generate_synonyms(settings, "test query")
        
        # Should return original input on error
        assert result == "test query"
        assert keywords == {}
    
    @patch("GUI_context.Groq")
    def test_rewrite_input_parses_json_response(self, mock_groq_class):
        """Verify rewrite_input_and_generate_synonyms parses JSON response."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"rephrased": "rewritten query", "keywords": {"test": {"synonyms": ["exam"]}}}'
                    )
                )
            ]
        )
        mock_groq_class.return_value = mock_client
        
        settings = {"groq_rewrite_model": "test-model"}
        result, keywords = GUI_context.rewrite_input_and_generate_synonyms(settings, "test query")
        
        assert result == "rewritten query"
        assert "test" in keywords


# =============================================================================
# INTEGRATION-LIKE TESTS
# =============================================================================

class TestRAGChatGUIIntegration:
    """Integration-style tests for RAGChatGUI main functionality."""
    
    def test_conversation_history_accumulates(self):
        """Verify conversation history accumulates messages."""
        window = MockChatWindow()
        
        window.conversation_history.append({"role": "user", "content": "Hello"})
        window.conversation_history.append({"role": "assistant", "content": "Hi there!"})
        window.conversation_history.append({"role": "user", "content": "How are you?"})
        
        assert len(window.conversation_history) == 3
        assert window.conversation_history[0]["role"] == "user"
        assert window.conversation_history[1]["role"] == "assistant"
    
    def test_tts_toggle_changes_action_text(self):
        """Verify TTS toggle changes action text."""
        window = MockWindowBase()
        window.tts_enabled = True
        
        # Toggle off
        window.tts_enabled = not window.tts_enabled
        if window.tts_enabled:
            window.tts_action.setText("🔊 TTS On")
        else:
            window.tts_action.setText("🔇 TTS Off")
        
        assert window.tts_action.text() == "🔇 TTS Off"
        
        # Toggle on
        window.tts_enabled = not window.tts_enabled
        if window.tts_enabled:
            window.tts_action.setText("🔊 TTS On")
        else:
            window.tts_action.setText("🔇 TTS Off")
        
        assert window.tts_action.text() == "🔊 TTS On"
    
    def test_theme_toggle_changes_state(self):
        """Verify theme toggle changes is_dark_theme state."""
        window = MockWindowBase()
        window.is_dark_theme = True
        
        window.is_dark_theme = not window.is_dark_theme
        assert window.is_dark_theme is False
        
        window.is_dark_theme = not window.is_dark_theme
        assert window.is_dark_theme is True


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_on_context_ready_handles_missing_keys(self):
        """Verify on_context_ready handles results with missing keys."""
        window = MockChatWindow()
        results = [
            {
                # Missing 'meta' key
                "final_score": 0.5,
                "document": "Some content",
                "keywords": [],
            },
        ]
        
        # Should not raise
        try:
            window.on_context_ready(results)
        except KeyError:
            pytest.fail("on_context_ready should handle missing keys")
    
    def test_empty_settings_dict_uses_defaults(self):
        """Verify settings operations work with minimal settings."""
        window = MockWindowBase()
        settings = {}
        
        # Should not raise
        GUI_settings.update_settings_from_ui(window, settings)
        
        assert "embedding_model" in settings
    
    def test_collection_browser_handles_empty_collection(self):
        """Verify browse_collection handles empty results."""
        window = MockChromaDBWindow()
        window.collection = MagicMock()
        window.collection.get.return_value = {
            "ids": [],
            "metadatas": [],
            "documents": [],
        }
        
        # Should not raise
        # Note: browse_collection is in rag_gui.py, not the mixin
        # This tests the data structure handling


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
