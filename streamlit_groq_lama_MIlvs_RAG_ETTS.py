import streamlit as st
import threading
import queue
import re
import os
import io
from pydub import AudioSegment
from pydub.playback import play
import asyncio
from groq import Groq
import ollama
import edge_tts
from pymilvus import connections, Collection

# Constants
EMBEDDINGS_DIR = "Embeddings"
MODEL = "mxbai-embed-large"
GROQ_MODEL = "llama-3.1-70b-versatile"
OLLAMA_MODEL = "phi-3"
JSON_FILE = "semantic_vault.json"

# ANSI escape codes for colors
PINK = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
NEON_GREEN = "\033[92m"
MAGENTA = "\033[35m"
BLUE = "\033[94m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET_COLOR = "\033[0m"

# Queue for audio playback
TTS_Audio_play_queue = queue.Queue()


# Function to process the queue
def process_TTS_Audio_play_queue(TTS_Audio_play_queue):
    while True:
        audio_fp = TTS_Audio_play_queue.get()
        if audio_fp is None:
            break
        audio_fp.seek(0)
        sound = AudioSegment.from_file(audio_fp, format="mp3")
        play(sound)
        TTS_Audio_play_queue.task_done()


# Start the worker thread
worker_thread = threading.Thread(
    target=process_TTS_Audio_play_queue, args=(TTS_Audio_play_queue,), daemon=True
)
worker_thread.start()


# Function to convert text to speech using edge-tts and play using pydub with speed adjustment
async def text_to_speech(text, speed=1.2, volume=1, voice="en-GB-MiaNeural"):
    try:
        rate = "+" + str(int((speed - 1) * 100)) + "%"
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        audio_bytes = b""

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        if not audio_bytes:
            raise ValueError(
                "No audio was received. Please verify that your parameters are correct."
            )

        audio_fp = io.BytesIO(audio_bytes)
        audio_fp.seek(0)

        # Adjust volume
        sound = AudioSegment.from_file(audio_fp, format="mp3")
        sound += 10 * (volume - 1)  # Adjust volume by increasing/decreasing decibels
        audio_fp = io.BytesIO()
        sound.export(audio_fp, format="mp3")
        audio_fp.seek(0)

        TTS_Audio_play_queue.put(audio_fp)

    except Exception as e:
        st.error(f"Error occurred during TTS: {e}")


# Function to process the queue
def process_TTS_queue(TTS_queue):
    while True:
        sentence = TTS_queue.get()
        if sentence is None:  # Sentinel value to stop the worker
            break
        asyncio.run(text_to_speech(sentence, speed=1.2))
        TTS_queue.task_done()


def chat_with_model(
    user_input,
    system_message,
    groq_model,
    ollama_model,
    conversation_history,
):
    try:
        response = groq_chat(
            user_input, system_message, groq_model, conversation_history
        )
    except Exception as e:
        if "Groq API limit reached" in str(e):
            response = ollama_chat(
                user_input, system_message, ollama_model, conversation_history
            )
        else:
            raise e

    return response


# Function to interact with the Ollama model
def ollama_chat(
    user_input,
    system_message,
    ollama_model,
    conversation_history,
):
    # Get relevant context from Milvus
    relevant_context = get_relevant_context(user_input, top_k=5)
    if relevant_context:
        print(
            "Context Pulled from Documents: \n\n"
            + CYAN
            + relevant_context
            + RESET_COLOR
            + "\n\n"
        )
    else:
        print("No relevant context found.")

    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = relevant_context + "\n\n" + user_input

    # Reset conversation history
    conversation_history = []
    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})

    # Create a message history including the system message and the conversation history
    messages = [{"role": "system", "content": system_message}, *conversation_history]

    # Send the completion request to the Ollama model with stream=True
    stream = ollama.chat(
        model=ollama_model,
        messages=messages,
        stream=True,
        keep_alive=-1,
    )

    # Queue for sentences
    TTS_queue = queue.Queue()

    # Start the worker thread
    worker_thread = threading.Thread(
        target=process_TTS_queue, args=(TTS_queue,), daemon=True
    )
    worker_thread.start()

    response = ""
    for chunk in stream:
        print(NEON_GREEN + chunk["message"]["content"], end="", flush=True)
        chunk_text = chunk["message"]["content"]
        response = f"{response}{chunk_text}"

        if any(delimiter in response for delimiter in ".;,!?"):
            response = response[1:]  # Remove the first character
            sentence, response = split_sentence(response)
            TTS_queue.put(sentence)

    # Print the response
    print(RESET_COLOR + "\n")

    return response


# Initialize Groq client
client = Groq(
    api_key="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",
)


def groq_chat(
    user_input,
    system_message,
    groq_model,
    conversation_history,
):
    # Get relevant context from Milvus
    relevant_context = get_relevant_context(user_input, top_k=5)
    if relevant_context:
        print(
            "Context Pulled from Documents: \n\n"
            + CYAN
            + relevant_context
            + RESET_COLOR
            + "\n\n"
        )
    else:
        print("No relevant context found.")

    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = relevant_context + "\n\n" + user_input

    # Reset conversation history
    conversation_history = []
    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})

    # Create a message history including the system message and the conversation history
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history,
        {"role": "user", "content": user_input},
    ]

    stream = client.chat.completions.create(
        #
        # Required parameters
        #
        messages=messages,
        # The language model which will generate the completion.
        model=groq_model,
        # Optional parameters
        # Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become deterministic
        # and repetitive.
        temperature=1,
        # The maximum number of tokens to generate. Requests can use up to
        # 2048 tokens shared between prompt and completion.
        max_tokens=7999,
        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=1,
        # A stop sequence is a predefined or user-specified text string that
        # signals an AI to stop generating content, ensuring its responses
        # remain focused and concise. Examples include punctuation marks and
        # markers like "[end]".
        stop="",
        # If set, partial message deltas will be sent.
        stream=True,
    )

    # Queue for sentences
    TTS_queue = queue.Queue()

    # Start the worker thread
    worker_thread = threading.Thread(
        target=process_TTS_queue, args=(TTS_queue,), daemon=True
    )
    worker_thread.start()

    response = ""
    response_chunks = []
    print(NEON_GREEN)
    for chunk in stream:
        chunk_text = chunk.choices[0].delta.content
        response = f"{response}{chunk_text}"
        # Update the text area with the current response
        response_chunks.append(chunk_text)
        print(chunk_text, end="", flush=True)

        if any(delimiter in response for delimiter in ".;,!?"):
            response = response[1:]  # Remove the first character
            sentence, response = split_sentence(response)
            TTS_queue.put(sentence)

    # Print the response
    print(RESET_COLOR + "\n")
    st.text_area("Response", value="".join(response_chunks), height=200)

    return response


def split_sentence(text):
    sentence_endings = ".;,!?"
    for ending in sentence_endings:
        if ending in text:
            parts = text.split(ending, 1)
            sentence = parts[0] + ending
            remainder = parts[1].strip()
            return sentence, remainder
    return text, ""


def get_relevant_context(rewritten_input, top_k=5):
    relevant_context = ""
    # Connect to Milvus
    connections.connect(host="localhost", port="19530")
    collection = Collection(name="html_chunks")

    # collection.load()

    # Encode the rewritten input
    input_embedding = ollama.embeddings(
        model=MODEL,
        prompt=rewritten_input,
        keep_alive=-1,
    )["embedding"]

    # Perform similarity search
    search_result = collection.search(
        data=[input_embedding],
        anns_field="embedding",
        param={
            "metric_type": "IP",
            "params": {},  # Search parameters
        },  # Search parameters
        limit=5,
        output_fields=["text"],  # Fields to return in the search results
        consistency_level="Bounded",
    )

    len(search_result)

    all_titles = [hit.entity.get("text") for hit in search_result[0]]
    # Assuming you have a list of titles in `all_titles`
    relevant_context = "\n\n".join(all_titles)

    """
    # Extract top_k results
    top_k_results = search_result[0]  # Assuming single query
    relevant_context = [result.entity["text"] for result in top_k_results]
    """

    # Disconnect from Milvus
    connections.disconnect(alias="html_chunks")

    return relevant_context


# Streamlit UI
st.title("Semantic Chatbot")
st.write("Welcome to the Semantic Chatbot. Please enter your message below.")

# Get user input
user_input = st.text_input("Enter your message:")

# Add sliders for speech speed and volume
speed = st.slider("Speech Speed", min_value=0.5, max_value=2.0, step=0.1, value=1.2)
volume = st.slider("Volume", min_value=0.5, max_value=2.0, step=0.1, value=1.0)

# Define system message
system_message = "You are a helpful assistant."

# Conversation history
conversation_history = []

if user_input:
    with st.spinner("Generating response..."):
        try:
            response = chat_with_model(
                user_input,
                system_message,
                GROQ_MODEL,
                OLLAMA_MODEL,
                conversation_history,
            )
            st.text_area("Response", value=response, height=200)

            # Display conversation history
            st.write("Conversation History:")
            for msg in conversation_history:
                role = msg["role"].capitalize()
                content = msg["content"]
                st.write(f"{role}: {content}")

            # Play TTS
            asyncio.run(text_to_speech(response, speed=speed, volume=volume))

        except Exception as e:
            st.error(f"Error occurred during chat: {e}")

# Ensure the worker thread shuts down cleanly when the app is closed
TTS_Audio_play_queue.put(None)  # Sentinel value to stop the worker
worker_thread.join()
