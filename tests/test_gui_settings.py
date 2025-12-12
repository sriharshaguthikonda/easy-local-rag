import json

import GUI_settings
from GUI_chat import ChatFunctionalityMixin


class DummyTextDisplay:
    def __init__(self):
        self.last_html = ""
        self.cleared = False

    def clear(self):
        self.cleared = True

    def setHtml(self, html):  # noqa: N802 (Qt-style)
        self.last_html = html


class DummyStatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message, timeout=None):  # noqa: N802 (Qt-style)
        self.messages.append((message, timeout))


class DummyCombo:
    def __init__(self, text=""):
        self._text = text

    def currentText(self):
        return self._text

    def setCurrentText(self, text):
        self._text = text


class DummyLineEdit:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text

    def setText(self, text):
        self._text = text


class DummySpin:
    def __init__(self, value=0):
        self._value = value

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value


class DummyTextEdit:
    def __init__(self, text=""):
        self._text = text

    def toPlainText(self):
        return self._text

    def setText(self, text):
        self._text = text


class DummyWindow:
    """Lightweight stand-in for the real PyQt window used by GUI_settings."""

    def __init__(
        self,
        embedding_model="embed",
        groq_model="groq",
        groq_rewrite_model="rewrite",
        ollama_model="ollama",
        chromadb_path="/tmp/chroma",
        collection="collection",
        system_message="hi",
        top_k=3,
        alpha=0.1,
        beta=0.2,
        gamma=0.3,
        lambda_mmr=0.4,
        tts_speed=1.1,
        tts_volume=0.9,
        tts_lang="en",
        tts_tld="com",
    ):
        self.embedding_model_combo = DummyCombo(embedding_model)
        self.groq_model_combo = DummyCombo(groq_model)
        self.groq_rewrite_combo = DummyCombo(groq_rewrite_model)
        self.ollama_model_combo = DummyCombo(ollama_model)
        self.chromadb_path_edit = DummyLineEdit(chromadb_path)
        self.collection_combo = DummyCombo(collection)
        self.system_message_edit = DummyTextEdit(system_message)
        self.top_k_spin = DummySpin(top_k)
        self.alpha_spin = DummySpin(alpha)
        self.beta_spin = DummySpin(beta)
        self.gamma_spin = DummySpin(gamma)
        self.lambda_mmr_spin = DummySpin(lambda_mmr)
        self.tts_speed_spin = DummySpin(tts_speed)
        self.tts_volume_spin = DummySpin(tts_volume)
        self.tts_lang_combo = DummyCombo(tts_lang)
        self.tts_tld_combo = DummyCombo(tts_tld)
        self.statusBar = DummyStatusBar()


class DummyChatWindow(ChatFunctionalityMixin):
    """Minimal stand-in to exercise mixin methods without Qt widgets."""

    def __init__(self):
        self.context_display = DummyTextDisplay()

    def append_message(self, *args, **kwargs):  # pragma: no cover - not needed here
        raise NotImplementedError

    def statusBar(self):  # pragma: no cover - placeholder
        return None


def test_update_settings_from_ui_reads_widget_values():
    window = DummyWindow()
    settings = GUI_settings.DEFAULT_SETTINGS.copy()

    # Adjust widget values after construction to verify live reads
    window.embedding_model_combo.setCurrentText("em1")
    window.groq_model_combo.setCurrentText("gm2")
    window.groq_rewrite_combo.setCurrentText("gr3")
    window.ollama_model_combo.setCurrentText("om4")
    window.chromadb_path_edit.setText("/data/chroma")
    window.collection_combo.setCurrentText("col5")
    window.system_message_edit.setText("system")
    window.top_k_spin.setValue(9)
    window.alpha_spin.setValue(0.6)
    window.beta_spin.setValue(0.7)
    window.gamma_spin.setValue(0.8)
    window.lambda_mmr_spin.setValue(0.9)
    window.tts_speed_spin.setValue(1.9)
    window.tts_volume_spin.setValue(0.1)
    window.tts_lang_combo.setCurrentText("fr")
    window.tts_tld_combo.setCurrentText("co.uk")

    GUI_settings.update_settings_from_ui(window, settings)

    assert settings["embedding_model"] == "em1"
    assert settings["groq_model"] == "gm2"
    assert settings["groq_rewrite_model"] == "gr3"
    assert settings["ollama_model"] == "om4"
    assert settings["chromadb_path"] == "/data/chroma"
    assert settings["collection_name"] == "col5"
    assert settings["system_message"] == "system"
    assert settings["top_k"] == 9
    assert settings["alpha"] == 0.6
    assert settings["beta"] == 0.7
    assert settings["gamma"] == 0.8
    assert settings["lambda_mmr"] == 0.9
    assert settings["tts_speed"] == 1.9
    assert settings["tts_volume"] == 0.1
    assert settings["tts_lang"] == "fr"
    assert settings["tts_tld"] == "co.uk"


def test_save_and_load_settings_round_trip(tmp_path, monkeypatch):
    settings = GUI_settings.DEFAULT_SETTINGS.copy()
    window = DummyWindow(
        embedding_model="em1",
        groq_model="gm2",
        groq_rewrite_model="gr3",
        ollama_model="om4",
        chromadb_path="/data/chroma",
        collection="col5",
        system_message="system",
        top_k=9,
        alpha=0.6,
        beta=0.7,
        gamma=0.8,
        lambda_mmr=0.9,
        tts_speed=1.9,
        tts_volume=0.1,
        tts_lang="fr",
        tts_tld="co.uk",
    )

    settings_path = tmp_path / "rag_gui_settings.json"
    monkeypatch.setattr(GUI_settings, "_settings_path", lambda: settings_path)

    GUI_settings.save_settings(window, settings)

    assert settings_path.exists(), "Settings file should be written"
    with settings_path.open("r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["embedding_model"] == "em1"

    # Change widget values to prove load_settings overwrites them
    window.embedding_model_combo.setCurrentText("different")
    window.groq_model_combo.setCurrentText("other-groq")
    window.system_message_edit.setText("changed")

    loaded_settings = GUI_settings.DEFAULT_SETTINGS.copy()
    GUI_settings.load_settings(window, loaded_settings)

    assert loaded_settings["embedding_model"] == "em1"
    assert window.embedding_model_combo.currentText() == "em1"
    assert loaded_settings["groq_model"] == "gm2"
    assert window.groq_model_combo.currentText() == "gm2"
    assert loaded_settings["system_message"] == "system"
    assert window.system_message_edit.toPlainText() == "system"


def test_on_context_ready_formats_results_html():
    window = DummyChatWindow()
    results = [
        {
            "meta": {"file_name": "doc1.txt"},
            "final_score": 0.42,
            "document": "Alpha keyword beta content",
            "keywords": ["keyword", "Alpha"],
        },
        {
            "meta": {"file_name": "doc2.txt"},
            "final_score": 0.9,
            "document": "Other text",
            "keywords": [],
        },
    ]

    window.on_context_ready(results)

    assert window.context_display.cleared is True
    html = window.context_display.last_html
    assert "doc1.txt" in html
    assert "Score: 0.420" in html
    assert '<span style="background-color: #FFD700; color: black;">Alpha</span>' in html
    assert '<span style="background-color: #FFD700; color: black;">keyword</span>' in html
