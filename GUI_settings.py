import os
import sys
import json

from PyQt5.QtWidgets import QMessageBox


DEFAULT_SETTINGS = {
    'embedding_model': 'mxbai-embed-large',
    'groq_model': 'llama-3.3-70b-versatile',
    'groq_rewrite_model': 'llama-3.1-8b-instant',
    'ollama_model': 'phi-3',
    'lmstudio_model': '',
    # Generation is deliberately scoped to one provider.  The model menus
    # remain separate so a stale Groq preference can never silently select an
    # Ollama model (or vice versa).
    'generation_provider': 'groq',
    'strict_generation_model': False,
    'generation_max_output_tokens': 1024,
    'collection_name': 'html_chunks_text_in_documents',
    'chromadb_path': r'C:\Windows_software\easy-local-rag\chroma',
    'system_message': 'You are a helpful assistant. You will give precise and concise answers from the given context. If the context does not have the answer then give it from your knowledge.',
    'top_k': 5,
    'additional_unique_files': 5,
    'alpha': 0.5,
    'beta': 0.3,
    'gamma': 0.2,
    'hybrid_alpha': 0.5,  # direct-search hybrid weights
    'hybrid_beta': 0.3,
    'hybrid_gamma': 0.2,
    'hybrid_delta': 0.2,  # sequential phrase boost
    'lambda_mmr': 0.5,
    'tts_speed': 1.4,
    'tts_volume': 0.5,
    'tts_lang': 'en',
    'tts_tld': 'co.uk',
    'just_search': False,
    'enter_to_send': False,
    'direct_search_mode': 'hybrid',  # hybrid | phrase
}


UNAVAILABLE_SUFFIX = " (unavailable)"
UNVERIFIED_SUFFIX = " (unverified)"


def normalize_model_id(value):
    """Remove legacy UI-only availability suffixes from a model ID."""
    model_id = str(value or "").strip()
    while model_id.endswith((UNAVAILABLE_SUFFIX, UNVERIFIED_SUFFIX)):
        model_id = model_id.rsplit(" (", 1)[0].rstrip()
    return model_id


def selected_model_id(combo):
    """Return raw user data instead of an availability suffix label."""
    try:
        data = combo.currentData()
        if data:
            return normalize_model_id(data)
    except (AttributeError, TypeError):
        pass
    return normalize_model_id(combo.currentText())


def update_settings_from_ui(window, settings):
    if window.embedding_model_combo.currentText():
        settings['embedding_model'] = selected_model_id(window.embedding_model_combo)
    if window.groq_model_combo.currentText():
        settings['groq_model'] = selected_model_id(window.groq_model_combo)
    if window.groq_rewrite_combo.currentText():
        settings['groq_rewrite_model'] = selected_model_id(window.groq_rewrite_combo)
    if window.ollama_model_combo.currentText():
        settings['ollama_model'] = selected_model_id(window.ollama_model_combo)
    provider_combo = getattr(window, 'generation_provider_combo', None)
    if provider_combo is not None:
        provider = (provider_combo.currentData()
                    if hasattr(provider_combo, 'currentData')
                    else None)
        provider = provider or provider_combo.currentText()
        if provider in {'groq', 'ollama', 'lmstudio'}:
            settings['generation_provider'] = provider
    strict_checkbox = getattr(window, 'strict_generation_model_checkbox', None)
    if strict_checkbox is not None:
        settings['strict_generation_model'] = bool(strict_checkbox.isChecked())
    settings['chromadb_path'] = window.chromadb_path_edit.text()
    settings['collection_name'] = window.collection_combo.currentText()
    settings['system_message'] = window.system_message_edit.toPlainText()
    settings['top_k'] = window.top_k_spin.value()
    settings['alpha'] = window.alpha_spin.value()
    settings['beta'] = window.beta_spin.value()
    settings['gamma'] = window.gamma_spin.value()
    settings['hybrid_alpha'] = getattr(window, "direct_alpha_spin", window.alpha_spin).value()
    settings['hybrid_beta'] = getattr(window, "direct_beta_spin", window.beta_spin).value()
    settings['hybrid_gamma'] = getattr(window, "direct_gamma_spin", window.gamma_spin).value()
    settings['hybrid_delta'] = getattr(window, "direct_delta_spin", window.gamma_spin).value()
    settings['lambda_mmr'] = window.lambda_mmr_spin.value()
    settings['tts_speed'] = window.tts_speed_spin.value()
    settings['tts_volume'] = window.tts_volume_spin.value()
    settings['tts_lang'] = window.tts_lang_combo.currentText()
    settings['tts_tld'] = window.tts_tld_combo.currentText()
    if getattr(window, "direct_mode_combo", None):
        settings['direct_search_mode'] = window.direct_mode_combo.currentText().lower()
    if getattr(window, "enter_to_send_checkbox", None):
        settings['enter_to_send'] = window.enter_to_send_checkbox.isChecked()


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

        # Older builds persisted the display suffix when a provider was
        # unavailable. Normalize those values before restoring the selectors.
        for key in ('groq_model', 'groq_rewrite_model', 'ollama_model'):
            if key in settings:
                settings[key] = normalize_model_id(settings[key])

        window.embedding_model_combo.setCurrentText(settings['embedding_model'])
        window.groq_model_combo.setCurrentText(settings['groq_model'])
        window.groq_rewrite_combo.setCurrentText(settings['groq_rewrite_model'])
        window.ollama_model_combo.setCurrentText(settings['ollama_model'])
        provider_combo = getattr(window, 'generation_provider_combo', None)
        if provider_combo is not None:
            provider = settings.get('generation_provider', 'groq')
            if provider not in {'groq', 'ollama', 'lmstudio'}:
                provider = 'groq'
                settings['generation_provider'] = provider
            provider_combo.setCurrentText(provider)
        strict_checkbox = getattr(window, 'strict_generation_model_checkbox', None)
        if strict_checkbox is not None:
            strict_checkbox.setChecked(bool(settings.get('strict_generation_model', False)))
        window.chromadb_path_edit.setText(settings['chromadb_path'])
        window.system_message_edit.setText(settings['system_message'])
        window.top_k_spin.setValue(settings['top_k'])
        window.alpha_spin.setValue(settings['alpha'])
        window.beta_spin.setValue(settings['beta'])
        window.gamma_spin.setValue(settings['gamma'])
        if getattr(window, "direct_alpha_spin", None):
            window.direct_alpha_spin.setValue(settings.get('hybrid_alpha', settings['alpha']))
        if getattr(window, "direct_beta_spin", None):
            window.direct_beta_spin.setValue(settings.get('hybrid_beta', settings['beta']))
        if getattr(window, "direct_gamma_spin", None):
            window.direct_gamma_spin.setValue(settings.get('hybrid_gamma', settings['gamma']))
        if getattr(window, "direct_delta_spin", None):
            window.direct_delta_spin.setValue(settings.get('hybrid_delta', settings['gamma']))
        window.lambda_mmr_spin.setValue(settings['lambda_mmr'])
        window.tts_speed_spin.setValue(settings['tts_speed'])
        window.tts_volume_spin.setValue(settings['tts_volume'])
        window.tts_lang_combo.setCurrentText(settings['tts_lang'])
        window.tts_tld_combo.setCurrentText(settings['tts_tld'])
        if getattr(window, "direct_mode_combo", None):
            window.direct_mode_combo.setCurrentText(settings.get('direct_search_mode', 'hybrid').capitalize())

        if hasattr(window, "statusBar") and window.statusBar:
            window.statusBar.showMessage("Settings loaded!", 3000)
    except Exception as e:
        print(f"Failed to load settings: {e}")
