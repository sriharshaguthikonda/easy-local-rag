import os

import ollama
from groq import Groq


class ModelLoadingMixin:
    """Shared model loading helpers for Groq and Ollama."""

    def load_groq_models(self):
        """Dynamically load available models from Groq API."""
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
        """Dynamically load available models from Ollama."""
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
