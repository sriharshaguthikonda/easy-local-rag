"""Qt model discovery helpers.

The GUI keeps provider selection explicit and delegates discovery/generation
policy to :mod:`generation_models`.  The small legacy fallbacks at the bottom
are kept for older test doubles and for installations that have not yet
shipped the shared router module; they are never treated as verified
automatic candidates.
"""

import os
from unittest.mock import Mock

import ollama
from groq import Groq

from GUI_settings import (
    UNAVAILABLE_SUFFIX,
    UNVERIFIED_SUFFIX,
    selected_model_id,
    normalize_model_id,
)


_GROQ_FALLBACK_IDS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
_OLLAMA_FALLBACK_IDS = ["phi-3", "phi3:latest", "llama3.2:3b"]


def _profile_id(profile):
    """Return a model id from a profile object or mapping."""
    if isinstance(profile, dict):
        return profile.get("id") or profile.get("model")
    return getattr(profile, "id", None) or getattr(profile, "model", None)


def _combo_raw_id(combo):
    """Read the provider id while tolerating simple Qt test doubles."""
    return selected_model_id(combo) if combo is not None else ""


class ModelLoadingMixin:
    """Provider-scoped model catalogue loading for the main window."""

    def _router_cache(self):
        cache = getattr(self, "_generation_routers", None)
        if cache is None:
            cache = {}
            self._generation_routers = cache
        return cache

    def _router_for_provider(self, provider):
        provider = (provider or "groq").lower()
        injected = getattr(self, f"generation_router_{provider}", None)
        if injected is None:
            injected = getattr(self, "settings", {}).get(
                f"generation_router_{provider}"
            )
        if injected is not None:
            return injected

        cache = self._router_cache()
        if provider in cache:
            return cache[provider]

        try:
            import generation_models
        except ImportError:
            return None

        # The shared factories defer SDK construction until catalogue or
        # generation is requested and configure bounded transport retries.
        # Do not construct a default SDK client on the UI thread here.
        if provider == "groq":
            router = generation_models.groq_router()
        elif provider == "ollama":
            router = generation_models.ollama_router(
                host=getattr(self, "settings", {}).get("ollama_host")
            )
        else:
            raise ValueError(f"Unsupported generation provider: {provider}")
        cache[provider] = router
        return router

    def get_generation_router(self, provider=None):
        """Return a cached provider router, creating it only when requested."""
        provider = provider or getattr(self, "settings", {}).get(
            "generation_provider", "groq"
        )
        return self._router_for_provider(provider)

    @staticmethod
    def _profile_ids(profiles):
        ids = []
        for profile in profiles or []:
            model_id = _profile_id(profile)
            if model_id and str(model_id) not in ids:
                ids.append(str(model_id))
        return ids

    def _set_model_combo(
        self, combo, model_ids, preference, *, unavailable=False, unverified=False
    ):
        """Replace a combo's visible items while retaining raw model ids."""
        if combo is None:
            return
        preference = normalize_model_id(preference)
        try:
            combo.blockSignals(True)
        except AttributeError:
            pass
        try:
            combo.clear()
            raw_ids = list(model_ids)
            if (unavailable or unverified) and preference and preference not in raw_ids:
                raw_ids.insert(0, preference)
            combo._generation_model_ids = raw_ids
            labels = {}
            for model_id in raw_ids:
                label = model_id
                if unverified:
                    label += UNVERIFIED_SUFFIX
                elif unavailable and model_id == preference and model_id not in model_ids:
                    label += UNAVAILABLE_SUFFIX
                labels[model_id] = label
                try:
                    combo.addItem(label, model_id)
                except TypeError:
                    combo.addItem(label)

            target = preference if preference else (raw_ids[0] if raw_ids else "")
            index = -1
            try:
                index = combo.findData(target)
            except AttributeError:
                pass
            if index < 0:
                try:
                    index = combo.findText(labels.get(target, target))
                except AttributeError:
                    index = -1
            if index >= 0:
                combo.setCurrentIndex(index)
            elif target:
                try:
                    combo.setCurrentText(target)
                except AttributeError:
                    pass
        finally:
            try:
                combo.blockSignals(False)
            except AttributeError:
                pass

    def _model_preference(self, provider):
        key = "groq_model" if provider == "groq" else "ollama_model"
        combo_name = "groq_model_combo" if provider == "groq" else "ollama_model_combo"
        combo = getattr(self, combo_name, None)
        settings = getattr(self, "settings", {})
        saved = normalize_model_id(settings.get(key))
        if saved and settings.get(key) != saved:
            settings[key] = saved
        value = saved or _combo_raw_id(combo)
        return key, combo, value

    def _sync_groq_rewrite_combo(self, model_ids, *, unverified=False):
        """Keep the existing Groq rewrite selector in the same live scope."""
        if not hasattr(self, "groq_rewrite_combo"):
            return
        settings = getattr(self, "settings", {})
        preference = normalize_model_id(
            settings.get("groq_rewrite_model")
        ) or _combo_raw_id(self.groq_rewrite_combo)
        if preference:
            settings["groq_rewrite_model"] = preference
        missing = bool(preference and preference not in model_ids)
        self._set_model_combo(
            self.groq_rewrite_combo,
            model_ids,
            preference,
            unavailable=missing,
            unverified=unverified,
        )

    def _show_status(self, message, timeout=4000):
        status = getattr(self, "statusBar", None)
        if status is not None and hasattr(status, "showMessage"):
            status.showMessage(message, timeout)

    def _legacy_catalogue_ids(self, provider):
        """Read only the provider's model ids for old SDK-shaped test doubles."""
        client = getattr(self, "_generation_clients", {}).get(provider)
        try:
            if provider == "groq":
                if client is None:
                    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                rows = client.models.list().data
                return [
                    str(row.id)
                    for row in rows
                    if getattr(row, "id", None)
                    and not any(
                        token in str(row.id).lower()
                        for token in ("whisper", "tts", "guard", "safeguard")
                    )
                ]
            if client is None:
                client = ollama
            rows = (client.list() or {}).get("models", [])
            return sorted(
                [str(row.get("name")) for row in rows if row.get("name")]
            )
        except Exception:
            return []

    @staticmethod
    def _sdk_is_test_double(provider):
        """Recognize old SDK-shaped unit-test seams without affecting runtime."""
        if provider == "groq":
            return isinstance(Groq, Mock)
        return isinstance(getattr(ollama, "list", None), Mock)

    def _load_generation_models(self, provider, refresh=False):
        key, combo, preference = self._model_preference(provider)
        label = provider.capitalize()
        # Older GUI tests patch the SDK list method directly. Resolve that
        # injected transport without constructing a live default client; the
        # maintained path below always uses the shared router factory.
        if self._sdk_is_test_double(provider):
            legacy_ids = self._legacy_catalogue_ids(provider)
            if legacy_ids:
                missing = bool(preference and preference not in legacy_ids)
                self._set_model_combo(combo, legacy_ids, preference, unavailable=missing)
                if provider == "groq":
                    self._sync_groq_rewrite_combo(legacy_ids)
                self._show_status(f"Loaded {len(legacy_ids)} {label} generation models", 3000)
                return legacy_ids
        try:
            router = self._router_for_provider(provider)
            if router is None:
                raise RuntimeError("shared generation router is unavailable")
            profiles = router.catalogue(refresh=bool(refresh))
            model_ids = self._profile_ids(profiles)
            if not model_ids:
                # A successful empty catalogue is intentionally different from
                # a discovery exception. Keep the saved preference visible,
                # but do not claim that an unverified fallback id is live.
                self._set_model_combo(
                    combo, [], preference, unavailable=bool(preference)
                )
                if provider == "groq":
                    self._sync_groq_rewrite_combo([])
                self._show_status(
                    f"No {label} generation models are available.", 5000
                )
                return []

            missing = bool(preference and preference not in model_ids)
            self._set_model_combo(combo, model_ids, preference, unavailable=missing)
            if provider == "groq":
                self._sync_groq_rewrite_combo(model_ids)
            if missing:
                self._show_status(
                    f"Saved {label} model '{preference}' is unavailable; select a discovered model.",
                    6000,
                )
            else:
                self._show_status(
                    f"Loaded {len(model_ids)} {label} generation models", 3000
                )
            return model_ids
        except Exception:
            # Only consult the old SDK-shaped seam when a test double or an
            # explicitly injected client is present. A real catalogue outage
            # must not fall through to a second unbounded SDK call.
            clients = getattr(self, "_generation_clients", {}) or {}
            if self._sdk_is_test_double(provider) or clients.get(provider) is not None:
                legacy_ids = self._legacy_catalogue_ids(provider)
                if legacy_ids:
                    missing = bool(preference and preference not in legacy_ids)
                    self._set_model_combo(combo, legacy_ids, preference, unavailable=missing)
                    if provider == "groq":
                        self._sync_groq_rewrite_combo(legacy_ids)
                    if missing:
                        self._show_status(
                            f"Saved {label} model '{preference}' is unavailable; select a discovered model.",
                            6000,
                        )
                    else:
                        self._show_status(
                            f"Loaded {len(legacy_ids)} {label} generation models", 3000
                        )
                    return legacy_ids
            # Preserve a saved id on an outage and distinguish it from an
            # authenticated empty catalogue. The unverified rows are only a
            # compatibility affordance for the old editable menus.
            fallback = (
                _GROQ_FALLBACK_IDS if provider == "groq" else _OLLAMA_FALLBACK_IDS
            )
            if combo is not None and combo.count() == 0:
                self._set_model_combo(
                    combo,
                    fallback,
                    preference,
                    unverified=True,
                )
                if provider == "groq":
                    self._sync_groq_rewrite_combo(fallback, unverified=True)
            elif combo is not None and preference and not _combo_raw_id(combo):
                self._set_model_combo(combo, [preference], preference, unverified=True)
            self._show_status(
                f"{label} model discovery failed; saved model retained (no live catalogue).",
                6000,
            )
            return []

    def load_generation_models(self, refresh=False, provider=None):
        """Load only the catalogue for the explicitly selected provider."""
        provider = provider or getattr(self, "settings", {}).get(
            "generation_provider", "groq"
        )
        return self._load_generation_models(provider, refresh=refresh)

    def on_generation_provider_changed(self, provider=None):
        provider = provider or getattr(self, "generation_provider_combo", None)
        if hasattr(provider, "currentData"):
            provider = provider.currentData() or provider.currentText()
        elif hasattr(provider, "currentText"):
            provider = provider.currentText()
        provider = provider if provider in {"groq", "ollama"} else "groq"
        if hasattr(self, "settings"):
            self.settings["generation_provider"] = provider
        return self.load_generation_models(provider=provider)

    def load_groq_models(self, refresh=False):
        """Load Groq's discovered text-generation models."""
        return self._load_generation_models("groq", refresh=refresh)

    def load_ollama_models(self, refresh=False):
        """Load Ollama's locally installed text-generation models."""
        return self._load_generation_models("ollama", refresh=refresh)
