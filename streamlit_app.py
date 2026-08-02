import streamlit as st

RAG_BACKEND_IMPORT_ERROR = None
try:
    import streamlit_groq_lama_chromadb_RAG_ETTS as rag_backend
except Exception as backend_error:
    rag_backend = None
    RAG_BACKEND_IMPORT_ERROR = backend_error


def _require_rag_backend():
    if rag_backend is None:
        raise RuntimeError(
            "RAG backend failed to load. "
            f"Original error: {RAG_BACKEND_IMPORT_ERROR}"
        )
    return rag_backend


if rag_backend is not None:
    get_relevant_context_hybrid = rag_backend.get_relevant_context_hybrid
    model = rag_backend.model
    groq_model = rag_backend.groq_model
    ollama_model = rag_backend.ollama_model
    initialize_collection = rag_backend.initialize_collection
    text_to_speech_gtts = rag_backend.text_to_speech_gtts
else:
    def get_relevant_context_hybrid(*args, **kwargs):  # type: ignore
        _require_rag_backend()

    model = "mxbai-embed-large"
    groq_model = "llama-3.3-70b-versatile"
    ollama_model = "phi-3"

    def initialize_collection():  # type: ignore
        _require_rag_backend()

    async def text_to_speech_gtts(*args, **kwargs):  # type: ignore
        _require_rag_backend()


"""TODO :  the file links are not working. it will switch to new chat when i click on the link of the file or open path."""
"""TODO :  the file links are not working. it will switch to new chat when i click on the link of the file or open path."""
"""TODO :  the file links are not working. it will switch to new chat when i click on the link of the file or open path."""

collection_name = "html_chunks"


from groq import Groq
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import ollama
from dotenv import load_dotenv
import os
from datetime import datetime
import json
import asyncio
import queue
import threading
from pathlib import Path
import webbrowser
import plotly.express as px
import pandas as pd
from datetime import timedelta
import networkx as nx
from pyvis.network import Network
import tempfile
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from conversation_import import apply_uploaded_conversation
from rag_prompting import (
    CONTEXT_GUARD,
    build_guarded_context_block,
    build_numbered_sources,
    context_from_sources,
    validate_response_citations,
)
from token_budget import trim_messages_to_budget

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


def _tts_worker_loop(tts_queue):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        sentence = tts_queue.get()
        if sentence is None:
            break
        if sentence.strip():
            loop.run_until_complete(text_to_speech_gtts(sentence.strip()))
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


# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient(
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
if "collection" not in st.session_state:
    try:
        st.session_state.collection = initialize_collection()
    except Exception as e:
        st.session_state.collection = None
        st.session_state.collection_init_error = str(e)
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0
if "message_timestamps" not in st.session_state:
    st.session_state.message_timestamps = []
if "confidence_scores" not in st.session_state:
    st.session_state.confidence_scores = []
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = True
tts_resource = ensure_tts_worker()
st.session_state.tts_queue = tts_resource["queue"]
st.session_state.tts_worker = tts_resource["worker"]
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []
if "source_filters" not in st.session_state:
    st.session_state.source_filters = {"file_types": [], "date_range": None}
if "visualization_data" not in st.session_state:
    st.session_state.visualization_data = {"topics": {}, "sources": {}}
if "favorite_responses" not in st.session_state:
    st.session_state.favorite_responses = []
if "tags" not in st.session_state:
    st.session_state.tags = {}

# Configure Streamlit page
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


def retrieve_numbered_sources(prompt, **kwargs):
    joined_context, retrieved_sources = get_relevant_context_hybrid(prompt, **kwargs)
    numbered_sources = build_numbered_sources(retrieved_sources)
    structured_context = context_from_sources(numbered_sources)
    if joined_context != structured_context:
        raise ValueError("Retrieved context does not match structured sources")
    return structured_context, numbered_sources


def build_citation_prompt(prefix, prompt, numbered_sources):
    citation_instruction = (
        "Use [N] citations for every factual claim that comes from context. "
        "If context does not support a claim, say so."
    )
    formatted_context = build_guarded_context_block(numbered_sources)
    return (
        f"{prefix}\n\n{CONTEXT_GUARD}\n{citation_instruction}\n\n"
        f"{formatted_context}\n\nQuery: {prompt}"
    )


def process_chat_mode(prompt, mode, context_window):
    """Handle different chat modes"""
    if mode == "Focused Search":
        # Use more specific context with higher relevance threshold
        context, numbered_sources = retrieve_numbered_sources(
            prompt, top_k=3, alpha=0.9, beta=0.1
        )

        processed_input = build_citation_prompt(
            "Focusing on most relevant sources:",
            prompt,
            numbered_sources,
        )
        return context, processed_input, numbered_sources

    elif mode == "Brain Dump":
        # Get more diverse sources with lower relevance threshold
        context, numbered_sources = retrieve_numbered_sources(
            prompt,
            top_k=10,
            additional_unique_files=10,
            alpha=0.6,
            beta=0.4,
            lambda_mmr=0.7,
        )

        processed_input = build_citation_prompt(
            "Drawing from multiple sources:",
            prompt,
            numbered_sources,
        )
        return context, processed_input, numbered_sources

    elif mode == "Summary":
        # Get context and ask for a summary
        context, numbered_sources = retrieve_numbered_sources(prompt, top_k=5)
        processed_input = build_citation_prompt(
            "Please summarize the following context:",
            prompt,
            numbered_sources,
        )
        return context, processed_input, numbered_sources

    else:  # Standard mode
        context, numbered_sources = retrieve_numbered_sources(prompt, top_k=5)
        processed_input = build_citation_prompt("", prompt, numbered_sources)
        return context, processed_input, numbered_sources


def chat_with_model(
    user_input, system_message, temperature=0.7, mode="Standard", context_window=5
):
    try:
        # Process input based on chat mode and capture both context and processed input
        context, processed_input, metadata = process_chat_mode(
            user_input, mode, context_window
        )

        messages = [
            {"role": "system", "content": system_message},
            *st.session_state.conversation_history[-context_window:],
            {"role": "user", "content": processed_input},
        ]
        messages, was_trimmed = trim_messages_to_budget(
            messages,
            max_input_tokens=5000,
            model_name=groq_model,
        )
        if (
            not messages
            or messages[-1].get("role") != "user"
            or messages[-1].get("content") != processed_input
        ):
            raise ValueError("Evidence prompt was truncated to fit token budget")
        if was_trimmed:
            st.info("Context was trimmed to fit token budget.")

        # Try Groq first
        try:
            completion = groq_client.chat.completions.create(
                messages=messages,
                model=groq_model,
                temperature=temperature,
                stream=True,
            )

            response_placeholder = st.empty()
            full_response = ""
            current_sentence = ""

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    current_sentence += content

                    # Check for sentence endings
                    if any(delimiter in content for delimiter in ".!?"):
                        if st.session_state.tts_enabled and current_sentence.strip():
                            st.session_state.tts_queue.put(current_sentence.strip())
                        current_sentence = ""

                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except Exception as e:
            # Fallback to Ollama
            stream = ollama.chat(model=ollama_model, messages=messages, stream=True)

            response_placeholder = st.empty()
            full_response = ""
            current_sentence = ""

            for chunk in stream:
                if "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    full_response += content
                    current_sentence += content

                    # Check for sentence endings
                    if any(delimiter in content for delimiter in ".!?"):
                        if st.session_state.tts_enabled and current_sentence.strip():
                            st.session_state.tts_queue.put(current_sentence.strip())
                        current_sentence = ""

                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        validate_response_citations(full_response, metadata)

        # Update conversation history
        st.session_state.conversation_history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": full_response},
            ]
        )

        return context, full_response, metadata

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None, None


def create_source_network(sources):
    """Create interactive network visualization of source relationships"""
    net = Network(height="500px", bgcolor="#ffffff")
    # ...network creation logic...
    return net


def generate_word_cloud(text):
    """Generate word cloud from text"""
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
            st.error(f"File not found: {file_path}")
            return False

        if os.name == "nt":  # Windows
            os.startfile(file_path)
        else:  # Linux/Mac
            webbrowser.open(f"file://{file_path}")
        return True
    except Exception as e:
        st.error(f"Error opening file: {e}")
        return False


def render_source_documents(metadata, cited_ids):
    source_map = {meta.get("citation_id"): meta for meta in metadata}
    ids_to_render = cited_ids or sorted(source_map.keys())

    with st.expander("Source Documents 📚", expanded=False):
        for citation_id in ids_to_render:
            meta = source_map.get(citation_id)
            if not meta:
                continue
            file_path = Path(meta["file_name"])
            st.markdown(f"### [{citation_id}] {file_path.name}")

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
            st.markdown(meta["document"])
            st.markdown("---")


def main():
    # Create three columns: sources, main chat, and analytics
    sources_col, main_col, analytics_col = st.columns([1, 2, 1])
    if RAG_BACKEND_IMPORT_ERROR:
        st.warning(f"RAG backend import warning: {RAG_BACKEND_IMPORT_ERROR}")
    if st.session_state.get("collection_init_error"):
        st.warning(f"Collection initialization warning: {st.session_state.collection_init_error}")

    # Initialize all settings first in the sidebar
    with st.sidebar:
        st.title("Settings ⚙️")

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

        # TTS controls
        st.subheader("Text-to-Speech Settings")
        st.session_state.tts_enabled = st.toggle(
            "Enable Text-to-Speech",
            value=st.session_state.tts_enabled,
        )
        if st.session_state.tts_enabled:
            tts_speed = st.slider("TTS Speed", 0.5, 2.0, 1.4, 0.1)
            tts_volume = st.slider("TTS Volume", 0.0, 2.0, 1.0, 0.1)

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
                        tags = st.text_input(f"Tags for {source['file_name']}")
                        if st.button("Save Tags"):
                            st.session_state.tags[source["file_name"]] = tags.split(",")

                        # Mark as favorite
                        if st.button("⭐ Favorite"):
                            if source not in st.session_state.favorite_responses:
                                st.session_state.favorite_responses.append(source)

    with main_col:
        st.title("RAG Chat Assistant 🤖")

        # Chat interface
        chat_container = st.container()

        # Display conversation history
        # ...existing conversation history code...

        # Chat input
        prompt = st.chat_input(
            "Enter your message...",
            key=f"chat_input_{st.session_state.chat_input_key}",
        )

        if prompt:
            # Process chat input with already defined settings
            with st.chat_message("user"):
                st.markdown(prompt)
                st.session_state.message_timestamps.append(datetime.now())

            with st.chat_message("assistant"):
                with st.spinner(f"Thinking... ({chat_mode} mode) 🤔"):
                    context, full_response, metadata = chat_with_model(
                        prompt, system_message, temperature, chat_mode, context_window
                    )
                    # Display sources with clickable links
                    if metadata:
                        cited_ids = validate_response_citations(full_response, metadata)
                        if not cited_ids:
                            st.warning("No citations returned in response.")
                        render_source_documents(metadata, cited_ids)

    with analytics_col:
        st.title("Analytics 📊")

        # Conversation Analytics
        with st.expander("Conversation Metrics"):
            metrics = analyze_conversation_metrics()
            st.metric("Total Messages", metrics["total_messages"])
            st.metric("Avg Response Time", f"{metrics['avg_response_time']:.2f}s")

        # Source Usage Visualization
        with st.expander("Source Usage"):
            if st.session_state.current_sources:
                source_data = pd.DataFrame(st.session_state.current_sources)
                fig = px.pie(source_data, values="relevance_score", names="file_name")
                st.plotly_chart(fig)

        # Topic Analysis
        with st.expander("Topic Analysis"):
            if st.session_state.conversation_history:
                all_text = " ".join(
                    [msg["content"] for msg in st.session_state.conversation_history]
                )
                wordcloud = generate_word_cloud(all_text)
                st.image(wordcloud.to_array())

        # Source Network
        with st.expander("Source Network"):
            net = create_source_network(st.session_state.current_sources)
            # Save and display network
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=500)

    # Keyboard Shortcuts Help
    with st.sidebar.expander("Keyboard Shortcuts ⌨️"):
        st.markdown("""
        - `/` : Focus chat input
        - `Ctrl+T` : Toggle TTS
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

        uploaded_file = st.file_uploader("Import Conversation")
        if uploaded_file:
            apply_uploaded_conversation(uploaded_file, st.session_state, st.warning, st.success)

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

    # Add keyboard shortcut to toggle TTS
    st.markdown(
        """
        <script>
            document.addEventListener('keydown', function(e) {
                if (e.key === 't' && e.ctrlKey) {
                    const ttsToggle = document.querySelector('div[data-testid="stToggleButton"]');
                    if (ttsToggle) ttsToggle.click();
                }
            });
        </script>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
