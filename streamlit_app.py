import asyncio
import inspect
import json
import os
import queue
import threading
import webbrowser
from datetime import datetime
from html import escape
from pathlib import Path
from typing import NamedTuple

import streamlit as st
from dotenv import load_dotenv

from conversation_import import apply_uploaded_conversation
from generation_models import GenerationError, groq_router, ollama_router
from offline_search import open_local_collection, search_documents
from rag_prompting import (
    build_guarded_request,
    build_numbered_sources,
    context_from_sources,
    validate_response_citations,
)
from token_budget import trim_messages_to_budget


DEFAULT_MODELS = {"groq": "llama-3.3-70b-versatile", "ollama": "phi-3"}
GENERATION_LABELS = {"groq": "Groq", "ollama": "Ollama", "offline": "Offline search"}


def _require_rag_backend():
    import streamlit_groq_lama_chromadb_RAG_ETTS as rag_backend

    return rag_backend


def get_relevant_context_hybrid(prompt, **options):
    backend = _require_rag_backend()
    if backend.collection is None:
        backend.initialize_collection()
    return backend.get_relevant_context_hybrid(prompt, **options)


def _tts_worker_loop(tts_queue):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        sentence = tts_queue.get()
        if sentence is None:
            break
        if sentence.strip():
            loop.run_until_complete(_require_rag_backend().text_to_speech_gtts(sentence.strip()))
        tts_queue.task_done()
    loop.close()


@st.cache_resource
def get_tts_worker_resource():
    tts_queue = queue.Queue()
    worker = threading.Thread(target=_tts_worker_loop, args=(tts_queue,), daemon=True)
    worker.start()
    return {"queue": tts_queue, "worker": worker}


def ensure_tts_worker():
    resource = get_tts_worker_resource()
    if not resource["worker"].is_alive():
        resource["queue"] = queue.Queue()
        resource["worker"] = threading.Thread(
            target=_tts_worker_loop,
            args=(resource["queue"],),
            daemon=True,
        )
        resource["worker"].start()
    return resource


def _initialize_session():
    defaults = {
        "conversation_history": [],
        "chat_input_key": 0,
        "message_timestamps": [],
        "allow_cloud_speech": False,
        "current_sources": [],
        "current_citations": [],
        "favorite_responses": [],
        "tags": {},
        "generation_provider": "groq",
        "strict_generation_model": False,
        "generation_routers": {},
        "generation_job": None,
        "last_generation_error": None,
        "last_generation_warning": None,
        "last_model_status": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _session_router(provider):
    routers = st.session_state.setdefault("generation_routers", {})
    if provider not in routers:
        load_dotenv()
        routers[provider] = groq_router() if provider == "groq" else ollama_router()
    return routers[provider]


def _generation_settings(*, active=False):
    provider = st.selectbox(
        "Provider / mode", tuple(GENERATION_LABELS),
        format_func=GENERATION_LABELS.__getitem__, key="generation_provider",
        disabled=active,
    )
    if provider == "offline":
        return provider, None, None, False

    routers = st.session_state.setdefault("generation_routers", {})
    router = routers.get(provider) if active else _session_router(provider)
    key = f"generation_model_{provider}"
    saved = st.session_state.setdefault(key, DEFAULT_MODELS[provider])
    refresh = st.button(
        "Refresh model catalogue",
        key=f"refresh_models_{provider}",
        disabled=active,
    )
    missing_status = "unavailable"
    if active:
        catalogue = list(
            st.session_state.get(f"generation_catalogue_{provider}", [])
        )
    else:
        try:
            catalogue = [profile.id for profile in router.catalogue(refresh=refresh)]
            st.session_state[f"generation_catalogue_{provider}"] = list(catalogue)
            if not catalogue:
                st.warning("The provider returned an empty generation model catalogue.")
        except GenerationError as error:
            st.warning(str(error))
            catalogue = []
            missing_status = "unverified"

    def label(model_id):
        return model_id if model_id in catalogue else f"{model_id} (saved, {missing_status})"

    options = list(dict.fromkeys([saved, *catalogue]))
    preferred = st.selectbox(
        "Preferred generation model",
        options,
        format_func=label,
        key=f"model_menu_{provider}",
        disabled=active,
    )
    st.session_state[key] = preferred
    strict = st.checkbox("Strict model pin", key="strict_generation_model", disabled=active)
    return provider, router, preferred, strict


def _configure_page():
    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for styling
    st.markdown(
        """
<style>
    .stTextInput:focus {
        box-shadow: 0 0 3px #4CAF50;
    }
    .message-container {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .assistant-message {
        background-color: #f0f0f0;
    }
    .metadata {
        font-size: 0.8em;
        color: #666;
    }
    .copy-button {
        float: right;
    }
</style>
    """,
        unsafe_allow_html=True,
    )


def retrieve_numbered_sources(prompt, *, retrieve=None, **kwargs):
    lookup = retrieve if retrieve is not None else get_relevant_context_hybrid
    try:
        signature = inspect.signature(lookup)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        accepted = set(signature.parameters)
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    joined_context, retrieved_sources = lookup(prompt, **kwargs)
    numbered_sources = build_numbered_sources(retrieved_sources)
    structured_context = context_from_sources(numbered_sources)
    if joined_context != structured_context:
        raise ValueError("Retrieved context does not match structured sources")
    return structured_context, numbered_sources


CITATION_INSTRUCTION = (
    "Use [N] citations for every factual claim that comes from context. "
    "If context does not support a claim, say so."
)


def process_chat_mode(
    prompt,
    mode,
    *,
    retrieve=None,
    rewrite=True,
    router=None,
    strict=False,
    cancelled=None,
):
    """Choose retrieval settings and a trusted task instruction for the current mode."""
    prefixes = {
        "Focused Search": "Focusing on most relevant sources:",
        "Brain Dump": "Drawing from multiple sources:",
        "Summary": "Please summarize the following context:",
    }
    settings = {
        "Focused Search": {"top_k": 3, "alpha": 0.9, "beta": 0.1},
        "Brain Dump": {
            "top_k": 10, "additional_unique_files": 10, "alpha": 0.6,
            "beta": 0.4, "lambda_mmr": 0.7,
        },
    }
    retrieval_options = {
        "rewrite": rewrite,
        **settings.get(mode, {"top_k": 5}),
    }
    if router is not None:
        retrieval_options["router"] = router
    if strict:
        retrieval_options["strict"] = True
    if cancelled is not None:
        retrieval_options["cancelled"] = cancelled
    context, sources = retrieve_numbered_sources(
        prompt, retrieve=retrieve, **retrieval_options
    )
    return context, prefixes.get(mode, ""), sources


class _PreparedGeneration(NamedTuple):
    context: str
    metadata: list
    was_trimmed: bool
    error: GenerationError | None
    request: dict


def _prepare_generation(
    user_input,
    system_message,
    temperature,
    mode,
    context_window,
    *,
    provider,
    preferred,
    router,
    strict,
    retrieve,
    conversation_history,
    cancelled=None,
):
    """Retrieve evidence and build the bounded request exactly once."""

    selected = preferred if preferred is not None else DEFAULT_MODELS[provider]
    context, task_instruction, metadata = process_chat_mode(
        user_input,
        mode,
        retrieve=retrieve,
        rewrite=provider == "groq",
        router=router,
        strict=strict,
        cancelled=cancelled,
    )
    history = list(conversation_history)[-context_window:] if context_window else []
    prepared = build_guarded_request(
        system_message, user_input, metadata, history,
        task_instruction=task_instruction, citation_instruction=CITATION_INSTRUCTION,
    )
    system_turn, current_turn = dict(prepared[0]), dict(prepared[-1])
    messages, was_trimmed = trim_messages_to_budget(
        prepared, max_input_tokens=5000, model_name=selected
    )
    error = None
    if len(messages) < 2 or messages[0] != system_turn or messages[-1] != current_turn:
        error = GenerationError(
            "input_budget", "The full evidence does not fit the input budget."
        )
    request = {
        "preferred": selected,
        "messages": messages,
        "strict": strict,
        "stream": True,
        "text_only": True,
        "temperature": temperature,
    }
    return _PreparedGeneration(context, metadata, was_trimmed, error, request)


def _accept_answer(placeholder, result, metadata, user_input, provider):
    """Validate citations before adding the answer to the chat or speech queue."""
    try:
        cited_ids = validate_response_citations(result.text, metadata)
    except ValueError:
        placeholder.empty()
        return "The answer cited an unknown source."

    status = f"{provider.capitalize()} · Used {result.actual_model}"
    if result.reason:
        status += f" · Preferred {result.preferred_model} · {result.reason}"
    state = st.session_state
    state.last_model_status = status
    state.current_citations = cited_ids
    state.last_generation_warning = (
        "No citations returned in response." if result.text and not cited_ids else None
    )
    st.caption(status)
    placeholder.text(result.text)
    state.conversation_history.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": result.text},
    ])
    if provider == "groq" and state.get("allow_cloud_speech") is True:
        ensure_tts_worker()["queue"].put(result.text)
    return None


def chat_with_model(
    user_input, system_message, temperature=0.7, mode="Standard", context_window=5,
    *, provider="groq", preferred=None, router=None, strict=False, retrieve=None,
    cancelled=None,
):
    st.session_state.current_sources = []
    st.session_state.current_citations = []
    selected_router = router if router is not None else _session_router(provider)
    try:
        prepared = _prepare_generation(
            user_input,
            system_message,
            temperature,
            mode,
            context_window,
            provider=provider,
            preferred=preferred,
            router=selected_router,
            strict=strict,
            retrieve=retrieve,
            conversation_history=st.session_state.conversation_history,
            cancelled=cancelled,
        )
    except Exception:
        st.error("No usable sources were retrieved for this request.")
        return None, None, None

    context, metadata = prepared.context, prepared.metadata
    st.session_state.current_sources = metadata
    if prepared.error is not None:
        st.error("The full evidence does not fit the input budget.")
        return context, None, metadata
    if prepared.was_trimmed:
        st.info("Earlier messages were trimmed to fit the input budget.")

    placeholder = st.empty()
    chunks = []

    def display_chunk(chunk):
        chunks.append(chunk)
        placeholder.text("".join(chunks) + "▌")

    try:
        result = selected_router.generate(
            **prepared.request, cancelled=cancelled, on_chunk=display_chunk
        )
    except GenerationError as error:
        if chunks:
            placeholder.text("".join(chunks))
        st.error(str(error))
        return context, None, metadata

    error = _accept_answer(placeholder, result, metadata, user_input, provider)
    if error:
        st.error(error)
        return context, None, metadata
    if st.session_state.last_generation_warning:
        st.warning(st.session_state.last_generation_warning)
    return context, result.text, metadata


class GenerationJob:
    """One Streamlit turn. The worker publishes chunks and its outcome only."""

    def __init__(self, question, provider):
        self.question = question
        self.provider = provider
        self.cancel_event = threading.Event()
        self.lock = threading.Lock()
        self.chunks = []
        self.outcome = None
        self.thread = None

    def append_chunk(self, chunk):
        with self.lock:
            self.chunks.append(chunk)

    def snapshot(self):
        with self.lock:
            return "".join(self.chunks), self.outcome


def _cancelled_error():
    return GenerationError("cancelled", "Generation cancelled.")


def _run_generation_job(job, router, **request):
    """Run retrieval and generation off the Streamlit thread."""
    prepared = None
    result = None
    error = None
    try:
        if job.cancel_event.is_set():
            raise _cancelled_error()
        prepared = _prepare_generation(
            job.question, provider=job.provider, router=router,
            cancelled=job.cancel_event.is_set, **request,
        )
        if prepared.error:
            raise prepared.error
        if job.cancel_event.is_set():
            raise _cancelled_error()
        result = router.generate(
            **prepared.request, cancelled=job.cancel_event.is_set, on_chunk=job.append_chunk
        )
    except GenerationError as exc:
        error = exc
    except Exception:
        message = (
            "No usable sources were retrieved for this request."
            if prepared is None else "Generation failed."
        )
        error = GenerationError("generation_error", message)
    finally:
        with job.lock:
            if job.cancel_event.is_set():
                error = _cancelled_error()
            job.outcome = prepared, result, error


def _active_generation_job():
    return st.session_state.get("generation_job")


def start_generation_job(
    user_input, system_message, temperature=0.7, mode="Standard", context_window=5,
    *, provider="groq", preferred=None, router=None, strict=False, retrieve=None,
):
    """Start one non-blocking generation per Streamlit session."""
    if (active := _active_generation_job()) is not None:
        return active
    selected_router = router if router is not None else _session_router(provider)
    job = GenerationJob(user_input, provider)
    job.thread = threading.Thread(
        target=_run_generation_job,
        args=(job, selected_router),
        kwargs={
            "system_message": system_message,
            "temperature": temperature,
            "mode": mode,
            "context_window": context_window,
            "preferred": preferred,
            "strict": strict,
            "retrieve": retrieve,
            "conversation_history": list(st.session_state.conversation_history),
        },
        daemon=True,
        name="streamlit-generation",
    )
    st.session_state.generation_job = job
    st.session_state.last_generation_error = None
    st.session_state.last_generation_warning = None
    job.thread.start()
    return job


def request_generation_cancel(job=None):
    """Set the cancellation signal while a provider call is in progress."""
    job = job or _active_generation_job()
    if job is None:
        return False
    with job.lock:
        if job.outcome is not None:
            return False
        job.cancel_event.set()
        return True


def _generation_observer_impl():
    job = _active_generation_job()
    if job is None:
        return
    chunks, outcome = job.snapshot()
    placeholder = st.empty()
    if outcome is None:
        if chunks and not job.cancel_event.is_set():
            placeholder.text(chunks + "▌")
        st.button(
            "Cancel generation",
            key="cancel_active_generation",
            on_click=request_generation_cancel,
            disabled=job.cancel_event.is_set(),
        )
        st.caption("Cancellation requested." if job.cancel_event.is_set() else "Generation in progress…")
        return

    prepared, result, error = outcome
    state = st.session_state
    state.generation_job = None
    state.current_sources = prepared.metadata if prepared is not None else []
    state.current_citations = []
    try:
        state.last_generation_error = (
            str(error) if error else _accept_answer(
                placeholder, result, state.current_sources, job.question, job.provider
            )
        )
        if state.last_generation_error:
            st.error(state.last_generation_error)
    finally:
        # Restore the input and render the accepted chat and cited sources together.
        rerun = getattr(st, "rerun", None)
        if callable(rerun):
            rerun()


_fragment = getattr(st, "fragment", None)
_generation_observer = (
    _fragment(run_every=0.25)(_generation_observer_impl)
    if _fragment is not None else _generation_observer_impl
)


def _offline_sources(query):
    collection = open_local_collection("./chroma", "html_chunks_temp")
    matches = search_documents(collection, query)
    return build_numbered_sources([
        {
            **record["metadata"],
            "file_name": record["metadata"].get("file_name") or "unknown",
            "collection_id": record["collection_id"],
            "chunk_id": record["chunk_id"],
            "document": record["document"],
            "relevance_score": record["relevance_score"],
        }
        for record in matches
    ])


def generate_word_cloud(text):
    """Generate word cloud from text"""
    from wordcloud import WordCloud

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    return wordcloud


def analyze_conversation_metrics():
    """Analyze conversation patterns and metrics"""
    metrics = {
        "total_messages": len(st.session_state.conversation_history),
        "avg_response_time": 0,
        "topic_distribution": {},
        "source_usage": {},
    }
    return metrics


def open_file(path):
    """Safely open a file using the default system application"""
    try:
        # Convert path to proper URI format
        file_path = Path(path).resolve()
        if not file_path.exists():
            st.error("The selected file was not found.")
            return False

        if os.name == "nt":  # Windows
            os.startfile(file_path)
        else:  # Linux/Mac
            webbrowser.open(f"file://{file_path}")
        return True
    except Exception as e:
        st.error("The selected file could not be opened.")
        st.text(str(e))
        return False


def render_source_documents(metadata, cited_ids):
    source_map = {meta.get("citation_id"): meta for meta in metadata}
    ids_to_render = cited_ids or sorted(source_map.keys())

    with st.expander("Source Documents 📚", expanded=False):
        for citation_id in ids_to_render:
            meta = source_map.get(citation_id)
            if not meta:
                continue
            file_path = Path(meta.get("file_path") or meta["file_name"])
            st.text(f"[{citation_id}] {file_path.name}")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(
                    "📂 Open File",
                    key=f"open_{citation_id}_{hash(str(file_path))}",
                ):
                    open_file(file_path)
            with col2:
                if st.button(
                    "📋 Copy Path",
                    key=f"copy_{citation_id}_{hash(str(file_path))}",
                ):
                    st.clipboard.write(str(file_path))

            st.markdown("**Excerpt:**")
            st.text(meta["document"])
            st.markdown("---")


def main():
    _configure_page()
    _initialize_session()
    active_job = _active_generation_job()
    sources_col, main_col, analytics_col = st.columns([1, 2, 1])

    # Initialize all settings first in the sidebar
    with st.sidebar:
        st.title("Settings ⚙️")
        provider, router, preferred, strict = _generation_settings(
            active=active_job is not None
        )
        if provider != "offline" and st.session_state.last_model_status:
            st.caption(f"Last answer · {st.session_state.last_model_status}")

        # System Message
        system_message = st.text_area(
            "System Message",
            "You are a helpful assistant. Give precise and concise answers from the context.",
            help="Define the AI's persona and behavior",
        )

        # Chat Mode Settings
        st.markdown("### Chat Mode Settings")
        chat_mode = st.selectbox(
            "Chat Mode",
            ["Standard", "Focused Search", "Brain Dump", "Summary"],
            help="""
            - Standard: Regular chat with balanced context
            - Focused Search: More specific, targeted responses
            - Brain Dump: Broader context from more sources
            - Summary: Summarize information from sources
            """,
        )

        # Context Window
        context_window = st.slider(
            "Context Window", 1, 10, 5, help="Number of previous messages to consider"
        )

        # Model settings
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.7,
            help="Higher values make responses more creative",
        )

        if provider == "groq":
            st.subheader("Text-to-Speech Settings")
            st.session_state.allow_cloud_speech = st.toggle(
                "Allow cloud speech (Google)", value=st.session_state.allow_cloud_speech,
                help="Send accepted assistant answers to Google Text-to-Speech for playback.",
            )

    # Now handle the column content
    with sources_col:
        st.sidebar.title("Knowledge Base 📚")

        # Source Management
        with st.sidebar.expander("Source Filters", expanded=False):
            # File type filter
            file_types = st.multiselect("File Types", [".txt", ".pdf", ".md", ".html"])

            # Date range filter
            date_range = st.date_input("Date Range", [])

            # Search within sources
            source_search = st.text_input("Search Sources")

        # Active Sources
        with st.sidebar.expander("Active Sources", expanded=True):
            # ...existing source display code...

            # Add source tagging
            if st.session_state.current_sources:
                for source in st.session_state.current_sources:
                    # ... existing source display ...
                    with st.expander("Source Actions"):
                        # Add tags
                        st.text(source["file_name"])
                        tags = st.text_input(
                            "Tags",
                            key=f"source_tags_{source.get('collection_id')}_{source.get('chunk_id')}_{source['citation_id']}",
                        )
                        if st.button("Save Tags"):
                            st.session_state.tags[source["file_name"]] = tags.split(",")

                        # Mark as favorite
                        if st.button("⭐ Favorite"):
                            if source not in st.session_state.favorite_responses:
                                st.session_state.favorite_responses.append(source)

    with main_col:
        st.title("RAG Chat Assistant 🤖")
        if active_job is None and st.session_state.get("last_generation_error"):
            st.error(st.session_state.last_generation_error)
            st.session_state.last_generation_error = None

        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.text(message["content"])
        if active_job is None and provider != "offline":
            if warning := st.session_state.get("last_generation_warning"):
                st.warning(warning)

        # Chat input
        prompt = st.chat_input(
            "Search local documents..." if provider == "offline" else "Enter your message...",
            key=f"chat_input_{st.session_state.chat_input_key}",
            disabled=active_job is not None,
        )

        metadata = None
        cited_ids = []
        if prompt:
            with st.chat_message("user"):
                st.text(prompt)
                st.session_state.message_timestamps.append(datetime.now())

            if provider == "offline":
                with st.chat_message("assistant"):
                    try:
                        metadata = _offline_sources(prompt)
                    except Exception:
                        st.error("The local search collection could not be read.")
                        metadata = []
                    else:
                        if not metadata:
                            st.info("No matching local documents.")
            elif active_job is None:
                active_job = start_generation_job(
                    prompt,
                    system_message,
                    temperature,
                    chat_mode,
                    context_window,
                    provider=provider,
                    preferred=preferred,
                    router=router,
                    strict=strict,
                )
                with st.chat_message("assistant"):
                    _generation_observer()
        elif active_job is not None:
            with st.chat_message("user"):
                st.text(active_job.question)
            with st.chat_message("assistant"):
                _generation_observer()

        if metadata:
            st.session_state.current_sources = metadata
            st.session_state.current_citations = []
            render_source_documents(metadata, cited_ids)
        elif _active_generation_job() is None and st.session_state.current_sources:
            render_source_documents(
                st.session_state.current_sources,
                st.session_state.get("current_citations", []),
            )

    with analytics_col:
        st.title("Analytics 📊")

        # Conversation Analytics
        with st.expander("Conversation Metrics"):
            metrics = analyze_conversation_metrics()
            st.metric("Total Messages", metrics["total_messages"])
            st.metric("Avg Response Time", f"{metrics['avg_response_time']:.2f}s")

        if provider != "offline":
            # Source Usage Visualization
            with st.expander("Source Usage"):
                if st.session_state.current_sources:
                    import pandas as pd
                    import plotly.express as px

                    source_data = pd.DataFrame(st.session_state.current_sources)
                    if {"score", "file_name"}.issubset(source_data.columns):
                        source_data["file_name"] = source_data["file_name"].map(escape)
                        fig = px.pie(source_data, values="score", names="file_name")
                        st.plotly_chart(fig)

            # Topic Analysis
            with st.expander("Topic Analysis"):
                if st.session_state.conversation_history:
                    all_text = " ".join(
                        [msg["content"] for msg in st.session_state.conversation_history]
                    )
                    wordcloud = generate_word_cloud(all_text)
                    st.image(wordcloud.to_array())

    # Keyboard Shortcuts Help
    with st.sidebar.expander("Keyboard Shortcuts ⌨️"):
        st.markdown("""
        - `/` : Focus chat input
        - `Ctrl+S` : Save current conversation
        - `Ctrl+F` : Search in conversation
        - `Ctrl+B` : Toggle source sidebar
        - `Esc` : Clear current input
        """)

    # Export/Import Conversations
    with st.sidebar.expander("Conversation Management"):
        if st.button("Export Conversation"):
            conversation_data = {
                "history": st.session_state.conversation_history,
                "sources": st.session_state.current_sources,
                "tags": st.session_state.tags,
                "favorites": st.session_state.favorite_responses,
            }
            st.download_button(
                "Download Conversation",
                data=json.dumps(conversation_data),
                file_name="conversation_export.json",
            )

        uploaded_file = st.file_uploader(
            "Import Conversation", disabled=active_job is not None
        )
        if active_job is not None:
            uploaded_file = None
        if uploaded_file:
            apply_uploaded_conversation(
                uploaded_file, st.session_state, st.warning, st.success
            )

    # Auto-focus script
    st.markdown(
        """
        <script>
            function focus_input() {
                const input = document.querySelector('.stChatInput input');
                if (input) input.focus();
            }
            document.addEventListener('keydown', (e) => {
                if (e.key === '/' && !e.ctrlKey && !e.altKey && !e.metaKey) {
                    e.preventDefault();
                    focus_input();
                }
            });
            focus_input();
        </script>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
