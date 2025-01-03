import streamlit as st
from groq_lama_chromadb_RAG_ETTS import (
    get_relevant_context_hybrid,
    groq_chat,
    ollama_chat,
    model,
    groq_model,
    ollama_model,
    collection_name,
    collection,  # Import the collection object
    initialize_collection,  # Import the initialization function
    text_to_speech_gtts,
    process_TTS_queue,
    TTS_Audio_play_queue,
)


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

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

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
    st.session_state.collection = initialize_collection()
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0
if "message_timestamps" not in st.session_state:
    st.session_state.message_timestamps = []
if "confidence_scores" not in st.session_state:
    st.session_state.confidence_scores = []
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = True
if "tts_queue" not in st.session_state:
    st.session_state.tts_queue = queue.Queue()
if "tts_worker" not in st.session_state:
    st.session_state.tts_worker = threading.Thread(
        target=process_TTS_queue, args=(st.session_state.tts_queue,), daemon=True
    )
    st.session_state.tts_worker.start()

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


def chat_with_model(user_input, system_message, temperature=0.7):
    try:
        # Get relevant context
        relevant_context = get_relevant_context_hybrid(user_input)

        user_input_with_context = (
            f"{relevant_context}\n\n{user_input}" if relevant_context else user_input
        )
        messages = [
            {"role": "system", "content": system_message},
            *st.session_state.conversation_history,
            {"role": "user", "content": user_input_with_context},
        ]

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
                            asyncio.run(text_to_speech_gtts(current_sentence.strip()))
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
                            asyncio.run(text_to_speech_gtts(current_sentence.strip()))
                        current_sentence = ""

                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        # Update conversation history
        st.session_state.conversation_history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": full_response},
            ]
        )

        return relevant_context, full_response

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None


def main():
    st.title("RAG Chat Assistant 🤖")

    # Sidebar settings
    with st.sidebar:
        st.title("Settings ⚙️")
        system_message = st.text_area(
            "System Message",
            "You are a helpful assistant...",
            help="Define the AI's persona and behavior",
        )

        # TTS controls
        st.subheader("Text-to-Speech Settings")
        st.session_state.tts_enabled = st.toggle("Enable Text-to-Speech", value=True)

        if st.session_state.tts_enabled:
            tts_speed = st.slider("TTS Speed", 0.5, 2.0, 1.4, 0.1)
            tts_volume = st.slider("TTS Volume", 0.0, 2.0, 1.0, 0.1)

        # Theme customization
        theme = st.selectbox(
            "Theme", ["Light", "Dark", "System"], help="Choose the interface theme"
        )

        # Model settings
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.7,
            help="Higher values make responses more creative",
        )

        # Clear conversation button with confirmation
        if st.button("Clear Conversation 🗑️"):
            if st.button("Are you sure? Click again to confirm"):
                st.session_state.conversation_history = []
                st.session_state.message_timestamps = []
                st.session_state.confidence_scores = []
                st.session_state.chat_input_key += 1
                st.experimental_rerun()

    # Main chat container
    chat_container = st.container()

    # Display conversation history with enhanced UI
    with chat_container:
        for idx, message in enumerate(st.session_state.conversation_history):
            role = message["role"]
            content = message["content"]

            # Get message metadata
            timestamp = (
                st.session_state.message_timestamps[idx]
                if idx < len(st.session_state.message_timestamps)
                else datetime.now()
            )
            confidence = (
                st.session_state.confidence_scores[idx]
                if idx < len(st.session_state.confidence_scores)
                else None
            )

            # Message container with metadata and controls
            with st.chat_message(role):
                st.markdown(content)

                # Metadata row
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(
                        f"<span class='metadata'>Sent: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</span>",
                        unsafe_allow_html=True,
                    )
                if confidence and role == "assistant":
                    with col2:
                        st.markdown(
                            f"<span class='metadata'>Confidence: {confidence:.2f}</span>",
                            unsafe_allow_html=True,
                        )
                with col3:
                    if st.button("Copy", key=f"copy_{idx}"):
                        st.write("Copied to clipboard!")
                        st.clipboard.write(content)

    # Chat input with auto-focus and character counter
    prompt = st.chat_input(
        "Enter your message...",
        key=f"chat_input_{st.session_state.chat_input_key}",
    )

    # Handle user input
    if prompt:
        # User message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.message_timestamps.append(datetime.now())

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                relevant_context, full_response = chat_with_model(
                    prompt, system_message, temperature
                )

                # Calculate simple confidence score based on context relevance
                confidence_score = (
                    len(relevant_context.split()) / 100 if relevant_context else 0.5
                )
                st.session_state.confidence_scores.append(confidence_score)

                # Display context in an expander
                if relevant_context:
                    with st.expander("View Source Context 📚", expanded=False):
                        st.markdown(relevant_context)
                        if st.button("Copy Context"):
                            st.clipboard.write(relevant_context)

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
