import os
import sys
import json

from PyQt5.QtWidgets import QMessageBox


DEFAULT_SETTINGS = {
    'embedding_model': 'mxbai-embed-large',
    'groq_model': 'llama-3.3-70b-versatile',
    'groq_rewrite_model': 'llama-3.1-8b-instant',
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


def update_settings_from_ui(window, settings):
    if window.embedding_model_combo.currentText():
        settings['embedding_model'] = window.embedding_model_combo.currentText()
    if window.groq_model_combo.currentText():
        settings['groq_model'] = window.groq_model_combo.currentText()
    if window.groq_rewrite_combo.currentText():
        settings['groq_rewrite_model'] = window.groq_rewrite_combo.currentText()
    if window.ollama_model_combo.currentText():
        settings['ollama_model'] = window.ollama_model_combo.currentText()
    settings['chromadb_path'] = window.chromadb_path_edit.text()
    settings['collection_name'] = window.collection_combo.currentText()
    settings['system_message'] = window.system_message_edit.toPlainText()
    settings['top_k'] = window.top_k_spin.value()
    settings['alpha'] = window.alpha_spin.value()
    settings['beta'] = window.beta_spin.value()
    settings['gamma'] = window.gamma_spin.value()
    settings['lambda_mmr'] = window.lambda_mmr_spin.value()
    settings['tts_speed'] = window.tts_speed_spin.value()
    settings['tts_volume'] = window.tts_volume_spin.value()
    settings['tts_lang'] = window.tts_lang_combo.currentText()
    settings['tts_tld'] = window.tts_tld_combo.currentText()


def _settings_path():
    base = os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv and sys.argv[0] else 'rag_gui.py'))
    return os.path.join(base, 'rag_gui_settings.json')


def save_settings(window, settings):
    update_settings_from_ui(window, settings)
    settings_path = _settings_path()
    try:
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        if hasattr(window, "statusBar") and window.statusBar:
            window.statusBar.showMessage("Settings saved!", 3000)
    except Exception as e:
        QMessageBox.warning(window, "Error", f"Failed to save settings: {e}")


def load_settings(window, settings):
    settings_path = _settings_path()
    if not os.path.exists(settings_path):
        return
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            settings.update(loaded)

        window.embedding_model_combo.setCurrentText(settings['embedding_model'])
        window.groq_model_combo.setCurrentText(settings['groq_model'])
        window.groq_rewrite_combo.setCurrentText(settings['groq_rewrite_model'])
        window.ollama_model_combo.setCurrentText(settings['ollama_model'])
        window.chromadb_path_edit.setText(settings['chromadb_path'])
        window.system_message_edit.setText(settings['system_message'])
        window.top_k_spin.setValue(settings['top_k'])
        window.alpha_spin.setValue(settings['alpha'])
        window.beta_spin.setValue(settings['beta'])
        window.gamma_spin.setValue(settings['gamma'])
        window.lambda_mmr_spin.setValue(settings['lambda_mmr'])
        window.tts_speed_spin.setValue(settings['tts_speed'])
        window.tts_volume_spin.setValue(settings['tts_volume'])
        window.tts_lang_combo.setCurrentText(settings['tts_lang'])
        window.tts_tld_combo.setCurrentText(settings['tts_tld'])

        if hasattr(window, "statusBar") and window.statusBar:
            window.statusBar.showMessage("Settings loaded!", 3000)
    except Exception as e:
        print(f"Failed to load settings: {e}")
