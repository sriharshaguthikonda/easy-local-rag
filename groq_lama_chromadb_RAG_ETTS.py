import argparse

import ollama
from groq import Groq

import httpcore  # Import httpcore to handle connection-related errors
import socket  # Import socket to handle network-related errors

import io
from pydub import AudioSegment
from pydub.playback import play

import threading
import subprocess

from pathlib import Path
from urllib.parse import urljoin


import queue
import re
import os
import time


import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

import speech_recognition as sr
import edge_tts
import pyttsx4  # Import pyttsx4 for offline TTS


import asyncio
import nest_asyncio

from dotenv import load_dotenv


from groq_lama_chromadb_RAG_ETTS_keyboard import handle_media_keys, control_flags

# Start the media key handler in a separate thread
threading.Thread(target=handle_media_keys, daemon=True).start()


# Constants
EMBEDDINGS_DIR = "Embeddings"
model = "mxbai-embed-large"
# groq_model="llama3-70b-8192"
groq_model = "llama-3.1-70b-versatile"
ollama_model = "llama3.2:latest"

collection_name = "html_chunks"

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


"""
######## ########  ######  
   ##       ##    ##    ## 
   ##       ##    ##       
   ##       ##     ######  
   ##       ##          ## 
   ##       ##    ##    ## 
   ##       ##     ######  
"""


# Apply the nest_asyncio patch
nest_asyncio.apply()

from threading import Event

# Add Events for playback control
pause_event = Event()  # Used to pause and resume playback
stop_event = Event()  # Used to stop playback


def process_TTS_Audio_play_queue():
    global control_flags
    while True:
        audio_fp = TTS_Audio_play_queue.get()

        # Check stop flag
        if control_flags["stop"]:
            TTS_Audio_play_queue.queue.clear()  # Clear the queue
            control_flags["stop"] = False
            stop_event.set()  # Signal to stop playback
            continue

        sound = AudioSegment.from_file(audio_fp, format="mp3")
        playback = threading.Thread(target=play, args=(sound,), daemon=True)
        playback.start()

        # Monitor for pause/resume/next while playback is active
        while playback.is_alive():
            if control_flags["pause"]:
                print("Playback paused")
                pause_event.clear()  # Block further playback
                pause_event.wait()  # Wait until resume is triggered

            if control_flags["next"]:
                print("Skipping to the next audio")
                control_flags["next"] = False
                break  # Break playback loop for the current file

        TTS_Audio_play_queue.task_done()


# Function to process the queue
def process_TTS_queue():
    while True:
        sentence = TTS_queue.get()
        if sentence is None:  # Sentinel value to stop the worker
            break
        asyncio.run(text_to_speech(sentence, speed=1.2))
        TTS_queue.task_done()


# Queue for sentences
TTS_Audio_play_queue = queue.Queue()

# Start the worker thread
threading.Thread(target=process_TTS_Audio_play_queue, daemon=True).start()

# Queue for sentences
TTS_queue = queue.Queue()

# Start the worker thread
threading.Thread(target=process_TTS_queue, daemon=True).start()


# Function to convert text to speech using edge-tts and play using pydub with speed adjustment
async def text_to_speech(text, speed=1.2, volume=1, voice="en-GB-MiaNeural"):
    try:
        # Online TTS using edge-tts
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

        # Place the audio data in the playback queue
        TTS_Audio_play_queue.put(audio_fp)

    except Exception as e:
        print(f"Error occurred during online TTS playback: {e}")
        print("Falling back to offline TTS...")
        fallback_offline_tts(text, speed, volume)


# Offline TTS fallback using pyttsx4
def fallback_offline_tts(text, speed=1, volume=1):
    try:
        # Initialize pyttsx4 engine
        engine = pyttsx4.init()

        # Set the speed (words per minute)
        engine.setProperty("rate", int(200 * speed))

        # Set the volume (0.0 to 1.0)
        engine.setProperty("volume", volume)

        # Speak the text
        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print(f"Offline TTS failed: {e}")


"""
 ######  ######## ######## 
##    ##    ##       ##    
##          ##       ##    
 ######     ##       ##    
      ##    ##       ##    
##    ##    ##       ##    
 ######     ##       ##    
"""


# Function to listen for the wake word
def listen_for_wake_word(wake_word="hey llama"):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for the wake word...")

        while True:
            audio = recognizer.listen(source)

            try:
                speech_text = recognizer.recognize_google(audio).lower()
                if wake_word in speech_text:
                    print(f"{wake_word.capitalize()} detected!")
                    prompt_user()
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("API unavailable")


"""
 ######  ##     ##    ###    ######## 
##    ## ##     ##   ## ##      ##    
##       ##     ##  ##   ##     ##    
##       ######### ##     ##    ##    
##       ##     ## #########    ##    
##    ## ##     ## ##     ##    ##    
 ######  ##     ## ##     ##    ##    
"""


def chat_with_model(
    user_input,
    system_message,
    groq_model,
    ollama_model,
    conversation_history,
):
    try:
        # Attempt to call Groq's model
        response = groq_chat(
            user_input, system_message, groq_model, conversation_history
        )
    except Exception as e:
        # Log the error and traceback for debugging
        print(f"Groq failed with error: {str(e)}")
        # Attempt to call Ollama's model
        response = ollama_chat(
            user_input, system_message, ollama_model, conversation_history
        )
        return response  # Return the response from Ollama

        # Check for specific known errors (connection issues)
        if isinstance(e, (httpcore.ConnectError, socket.gaierror)):
            print("Connection error with Groq detected. Redirecting to Ollama...")
            try:
                # Attempt to call Ollama's model
                response = ollama_chat(
                    user_input, system_message, ollama_model, conversation_history
                )
                return response  # Return the response from Ollama
            except Exception as ollama_error:
                # If Ollama also fails, log the error
                print(f"Ollama fallback failed with error: {str(ollama_error)}")

                raise ollama_error  # Raise the error for handling
        else:
            # If it's an unknown error, re-raise it
            raise e

    return response


"""
 #######  ##       ##          ###    ##     ##    ###    
##     ## ##       ##         ## ##   ###   ###   ## ##   
##     ## ##       ##        ##   ##  #### ####  ##   ##  
##     ## ##       ##       ##     ## ## ### ## ##     ## 
##     ## ##       ##       ######### ##     ## ######### 
##     ## ##       ##       ##     ## ##     ## ##     ## 
 #######  ######## ######## ##     ## ##     ## ##     ## 
"""


# Function to interact with the Ollama model
def ollama_chat(
    user_input,
    system_message,
    ollama_model,
    conversation_history,
):
    # Get relevant context from Milvus
    relevant_context = get_relevant_context(user_input, top_k=5)

    # Prepare the user's input by concatenating it with the relevant context
    if relevant_context:
        user_input_with_context = relevant_context + "\n\n" + user_input
    else:
        user_input_with_context = user_input

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

    response = ""

    current_sentence = ""
    word_buffer = ""  # To accumulate small chunks of words

    for chunk in stream:
        print(NEON_GREEN + chunk["message"]["content"], end="", flush=True)
        if not chunk.get("message") or not chunk["message"].get("content"):
            continue

        chunk_text = chunk["message"]["content"].replace(
            "\n", " "
        )  # Remove unintended newlines

        # Accumulate the chunk text
        word_buffer += chunk_text

        # Process the buffer when it contains a sentence-ending punctuation
        sentence_endings = ".!? "
        if word_buffer and any(word_buffer.endswith(end) for end in sentence_endings):
            current_sentence += word_buffer
            word_buffer = ""  # Reset buffer after processing

            # If a coherent chunk or sentence is formed, send it to TTS
            if any(current_sentence.endswith(end) for end in sentence_endings):
                # Strip markdown-like symbols for TTS
                stripped_text = re.sub(r"[\*_]", "", current_sentence)

                # Send text to TTS immediately
                if stripped_text:
                    TTS_queue.put(stripped_text)  # Send to TTS queue
                    current_sentence = ""  # Reset after sending to TTS

    # Print the response
    print(RESET_COLOR + "\n")

    return response


"""
 ######   ########   #######   #######  
##    ##  ##     ## ##     ## ##     ## 
##        ##     ## ##     ## ##     ## 
##   #### ########  ##     ## ##     ## 
##    ##  ##   ##   ##     ## ##  ## ## 
##    ##  ##    ##  ##     ## ##    ##  
 ######   ##     ##  #######   ##### ## 
"""

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(
    api_key=api_key,
)


def groq_chat(
    user_input,
    system_message,
    groq_model,
    conversation_history,
):
    # Get relevant context from Milvus
    relevant_context = get_relevant_context(user_input, top_k=5)

    # Prepare the user's input by concatenating it with the relevant context
    if relevant_context:
        user_input_with_context = relevant_context + "\n\n" + user_input
    else:
        user_input_with_context = user_input

    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})

    # Create a message history including the system message and the conversation history
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history,
        {"role": "user", "content": user_input},
    ]

    stream = client.chat.completions.create(
        # Required parameters
        messages=messages,
        model=groq_model,
        temperature=1,
        # The maximum number of tokens to generate. Requests can use up to
        # 2048 tokens shared between prompt and completion.
        max_tokens=7999,
        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=1,
        stop="",
        # If set, partial message deltas will be sent.
        stream=True,
    )

    response = ""

    current_sentence = ""
    word_buffer = ""  # To accumulate small chunks of words

    print(NEON_GREEN)
    for chunk in stream:
        print(chunk.choices[0].delta.content, end="")
        chunk_text = chunk.choices[0].delta.content

        if chunk_text is not None:
            # Accumulate the chunk text
            word_buffer += chunk_text

        # Process the buffer when it contains a sentence-ending punctuation
        sentence_endings = ".!?,"
        if word_buffer and any(word_buffer.endswith(end) for end in sentence_endings):
            current_sentence += word_buffer
            word_buffer = ""  # Reset buffer after processing

            # If a coherent chunk or sentence is formed, send it to TTS
            if any(current_sentence.endswith(end) for end in sentence_endings):
                # Strip markdown-like symbols for TTS
                stripped_text = re.sub(r"[\*_]", "", current_sentence)

                # Send text to TTS immediately
                if stripped_text:
                    TTS_queue.put(stripped_text)  # Send to TTS queue
                    current_sentence = ""  # Reset after sending to TTS

    # Print the response
    print(RESET_COLOR + "\n")

    return response


"""
 ######  ######## ##    ## ######## ######## ##    ##  ######  ######## 
##    ## ##       ###   ##    ##    ##       ###   ## ##    ## ##       
##       ##       ####  ##    ##    ##       ####  ## ##       ##       
 ######  ######   ## ## ##    ##    ######   ## ## ## ##       ######   
      ## ##       ##  ####    ##    ##       ##  #### ##       ##       
##    ## ##       ##   ###    ##    ##       ##   ### ##    ## ##       
 ######  ######## ##    ##    ##    ######## ##    ##  ######  ######## 
"""


# Define the function to split sentences with the condition
def split_sentence(response):
    delimiters = r"[.,;!?]"  # Delimiters for splitting
    # Split the response based on delimiters
    sentences = re.split(delimiters, response, maxsplit=1)

    # If we have more than one part after the split
    if len(sentences) > 1:
        sentence, remaining_response = sentences[0], sentences[1]

        # Check if the sentence has 10 words or more
        if len(sentence.split()) < 5:
            # If the sentence is too short, don't split
            sentence = sentence + remaining_response
            remaining_response = ""
    else:
        # No split happened, just return the original response
        sentence, remaining_response = sentences[0], ""

    return sentence.strip(), remaining_response.strip()


"""
 ######   #######  ##    ## ######## ######## ##     ## ######## 
##    ## ##     ## ###   ##    ##    ##        ##   ##     ##    
##       ##     ## ####  ##    ##    ##         ## ##      ##    
##       ##     ## ## ## ##    ##    ######      ###       ##    
##       ##     ## ##  ####    ##    ##         ## ##      ##    
##    ## ##     ## ##   ###    ##    ##        ##   ##     ##    
 ######   #######  ##    ##    ##    ######## ##     ##    ##    
"""


def get_relevant_context(rewritten_input, top_k=5):
    # global collection

    try:
        relevant_context = ""

        # Encode the rewritten input
        input_embedding = ollama.embeddings(
            model=model,
            prompt=rewritten_input,
            keep_alive=-1,
        )["embedding"]

        # Perform similarity search
        search_result = collection.query(
            query_embeddings=[input_embedding],
            n_results=top_k,
            include=["metadatas"],  # Fields to return in the search results
        )

        # Extract relevant context
        if search_result:
            all_titles = [item["text"] for item in search_result["metadatas"][0]]
            relevant_context = "\n\n".join(all_titles)

        worker_thread = threading.Thread(
            target=print_relevant_context,
            args=(search_result,),
            daemon=True,
        )

        worker_thread.start()

        return relevant_context

    except Exception as e:
        print(f"An error occurred: {e}")
        return "answer this yourself!"


def print_relevant_context(search_result):
    print("Context Pulled from Documents: \n\n")

    for item in search_result["metadatas"][0]:
        file_path = item["file_name"]
        # Replace backslashes in file_path to make it clickable using raw string
        clickable_file_path = urljoin("file:", Path(file_path).as_uri())
        print(
            YELLOW  # Yellow color
            + item["text"]
            + "\n"
            + BLUE  # Blue color
            + clickable_file_path
            + RESET_COLOR  # Reset color
            + "\n\n"
        )
        # Open the file in the default web browser
        # webbrowser.open(file_path)


"""
##          ###    ##     ##    ###        ######  ########    ###    ########  ######## 
##         ## ##   ###   ###   ## ##      ##    ##    ##      ## ##   ##     ##    ##    
##        ##   ##  #### ####  ##   ##     ##          ##     ##   ##  ##     ##    ##    
##       ##     ## ## ### ## ##     ##     ######     ##    ##     ## ########     ##    
##       ######### ##     ## #########          ##    ##    ######### ##   ##      ##    
##       ##     ## ##     ## ##     ##    ##    ##    ##    ##     ## ##    ##     ##    
######## ##     ## ##     ## ##     ##     ######     ##    ##     ## ##     ##    ##    
"""


def check_and_start_ollama():
    try:
        # Check if Ollama is serving by attempting to connect to the server
        response = subprocess.run(
            ["curl", "-I", "http://127.0.0.1:11434/"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Check if the response indicates the server is up
        if "HTTP/1.1 200 OK" in response.stdout:
            print("Ollama is already serving.")
        else:
            start_ollama_server()
    except subprocess.CalledProcessError:
        start_ollama_server()
    except Exception as e:
        print(f"An error occurred: {e}")


def start_ollama_server():
    # Redirect output to devnull to suppress it
    with open(os.devnull, "w") as devnull:
        subprocess.Popen(["ollama", "serve"], stdout=devnull, stderr=devnull)
        print("Ollama has been started.")
        print("Enter your message (or 'exit' to quit):")


"""
##     ##    ###    #### ##    ## 
###   ###   ## ##    ##  ###   ## 
#### ####  ##   ##   ##  ####  ## 
## ### ## ##     ##  ##  ## ## ## 
##     ## #########  ##  ##  #### 
##     ## ##     ##  ##  ##   ### 
##     ## ##     ## #### ##    ## 
"""


def main():
    global collection
    # Reset conversation history
    global conversation_history

    parser = argparse.ArgumentParser(description="Ollama Chat")
    parser.add_argument(
        "--model", default="phi3", help="Ollama model to use phi3 (default: llama3)"
    )
    args = parser.parse_args()

    # Example conversation history
    conversation_history = [{"role": "system", "content": "Welcome to Ollama Chat!"}]

    # Start the Ollama server in a separate thread
    ollama_thread = threading.Thread(target=check_and_start_ollama, daemon=True)
    ollama_thread.start()
    threading.Thread(target=handle_media_keys, daemon=True).start()

    client = chromadb.PersistentClient(
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    # Get or create the collection
    collection = client.get_collection(collection_name)

    while True:
        user_input = input(
            "\n"
            + RED
            + BOLD
            + "Enter your message (or 'exit' to quit):"
            + "\n"
            + RESET_COLOR
            + "\n"
        )
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break

        system_message = "You are a helpful assistant. You will give precise and concise answers from the given context. if the context doesnot have the answer then give it from your knowledge"

        # Interact with the Ollama model
        if user_input:
            chat_with_model(
                user_input,
                system_message,
                groq_model,
                ollama_model,
                conversation_history=conversation_history,
            )


if __name__ == "__main__":
    main()
