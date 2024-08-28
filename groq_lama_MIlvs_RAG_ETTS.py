import argparse

import ollama
from groq import Groq

import io
from pydub import AudioSegment
from pydub.playback import play

import threading
import subprocess

import queue
import re
import os
import time


from pymilvus import (
    MilvusClient,
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

import speech_recognition as sr
import edge_tts

import asyncio
import nest_asyncio

# Constants
EMBEDDINGS_DIR = "Embeddings"
model = "mxbai-embed-large"
# groq_model="llama3-70b-8192"
groq_model = "llama-3.1-70b-versatile"
ollama_model = "phi-3"
json_file = "semantic_vault.json"

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


# Function to process the queue
def process_TTS_Audio_play_queue(TTS_Audio_play_queue):
    while True:
        audio_fp = TTS_Audio_play_queue.get()
        audio_fp.seek(0)

        sound = AudioSegment.from_file(audio_fp, format="mp3")

        # Play the adjusted audio
        play(sound)
        TTS_Audio_play_queue.task_done()


# Queue for sentences
TTS_Audio_play_queue = queue.Queue()

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

        TTS_Audio_play_queue.put(audio_fp)

    except Exception as e:
        print(f"Error occurred during playback: {e}")


# Function to process the queue
def process_TTS_queue(TTS_queue):
    while True:
        sentence = TTS_queue.get()
        if sentence is None:  # Sentinel value to stop the worker
            break
        asyncio.run(text_to_speech(sentence, speed=1.2))
        TTS_queue.task_done()


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


"""
 ######   ########   #######   #######  
##    ##  ##     ## ##     ## ##     ## 
##        ##     ## ##     ## ##     ## 
##   #### ########  ##     ## ##     ## 
##    ##  ##   ##   ##     ## ##  ## ## 
##    ##  ##    ##  ##     ## ##    ##  
 ######   ##     ##  #######   ##### ## 
"""

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
    print(NEON_GREEN)
    for chunk in stream:
        print(chunk.choices[0].delta.content, end="")
        chunk_text = chunk.choices[0].delta.content
        response = f"{response}{chunk_text}"

        if any(delimiter in response for delimiter in ".;!?"):
            response = response[1:]  # Remove the first character
            sentence, response = split_sentence(response)
            TTS_queue.put(sentence)

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


# Define the function to split sentences
def split_sentence(response):
    # delimiters = r"[.,;!?]"  # Add more delimiters if needed
    delimiters = r"[\n]"  # Add more delimiters if needed
    sentences = re.split(delimiters, response, maxsplit=1)
    if len(sentences) > 1:
        sentence, response = sentences[0], sentences[1]
    else:
        sentence, response = sentences[0], ""
    return sentence, response


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
    try:
        relevant_context = ""

        # collection.load()

        # Encode the rewritten input
        input_embedding = ollama.embeddings(
            model=model,
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

        # len(search_result)
        if search_result:
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

    except Exception as e:
        print(f"An error occurred: {e}")
        return "answer this yourself!"


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
########   ######  ##    ## ########        ##     ## ##       ##     ##  ######         ######  ##     ## ##    ## 
##     ## ##    ## ##   ##  ##     ##       ###   ### ##       ##     ## ##    ##       ##    ## ##     ## ##   ##  
##     ## ##       ##  ##   ##     ##       #### #### ##       ##     ## ##             ##       ##     ## ##  ##   
##     ## ##       #####    ########        ## ### ## ##       ##     ##  ######        ##       ######### #####    
##     ## ##       ##  ##   ##   ##         ##     ## ##        ##   ##        ##       ##       ##     ## ##  ##   
##     ## ##    ## ##   ##  ##    ##        ##     ## ##         ## ##   ##    ##       ##    ## ##     ## ##   ##  
########   ######  ##    ## ##     ##       ##     ## ########    ###     ######         ######  ##     ## ##    ## 
"""


def is_docker_running():
    try:
        # Run the `docker info` command to check if Docker is running
        result = subprocess.run(
            ["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            print("Docker is running")
            return True
        else:
            print("Docker is not running")
            return False
    except FileNotFoundError:
        # Docker command not found
        return False


def is_Milvus_running():
    load_state_str = "None"
    try:
        Milvus_collection_Client = MilvusClient(uri="http://localhost:19530")
        collection_name = "html_chunks"
        collection_load_state = Milvus_collection_Client.get_load_state(collection_name)
        load_state_str = str(collection_load_state["state"])
        print("Milvus is", load_state_str)

        if load_state_str == "Loaded":
            return True
        elif load_state_str == "Loading":
            time.sleep(10)  # Wait 10 seconds and recheck
            return is_Milvus_running()
        elif load_state_str == "NotLoad":
            print(f"Collection {collection_name} is not loaded.")
            # Optionally, you can trigger a load operation here
            return False
        elif load_state_str == "NotExist":
            print(f"Collection {collection_name} does not exist.")
            # Handle the case where the collection does not exist
            return False
        else:
            print(f"Unknown load state: {load_state_str}")
            return False
    except Exception as e:
        print(f"Error checking Milvus state: {e}")
        return False


def start_docker_milvus():
    try:
        subprocess.run(
            ["docker-compose", "up", "-d"],
            cwd="C:\\Users\\deletable\\OneDrive\\easy-local-rag",
            check=True,
        )
        print("Milvus started using Docker.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Milvus with Docker: {e}")


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

    parser = argparse.ArgumentParser(description="Ollama Chat")
    parser.add_argument(
        "--model", default="phi3", help="Ollama model to use phi3 (default: llama3)"
    )
    args = parser.parse_args()

    # Example conversation history
    conversation_history = [{"role": "system", "content": "Welcome to Ollama Chat!"}]

    # Start the Ollama server in a separate thread
    ollama_thread = threading.Thread(target=check_and_start_ollama)
    ollama_thread.start()

    # Connect to Milvus
    connections.connect(host="localhost", port="19530")
    collection = Collection(name="html_chunks")

    if not is_docker_running():
        print("start docker")
    elif not is_Milvus_running():
        start_docker_milvus()

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
