import argparse
import ollama
import edge_tts
import io
from pydub import AudioSegment
from pydub.playback import play
import threading
import queue
import re
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

import asyncio
import nest_asyncio


# Constants
EMBEDDINGS_DIR = "Embeddings"
model = "mxbai-embed-large"
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
                     ####### #######  #####  
                        #       #    #     # 
                        #       #    #       
                        #       #     #####  
                        #       #          # 
                        #       #    #     # 
                        #       #     #####                           
"""

# Apply the nest_asyncio patch
nest_asyncio.apply()


# Function to process the queue
def process_TTS_Audio_play_queue(TTS_Audio_play_queue):
    while True:
        audio_fp = TTS_Audio_play_queue.get()
        audio_fp.seek(0)

        sound = AudioSegment.from_file(audio_fp, format="mp3")

        # Adjust the speed (150ms chunks with 25ms crossfade)
        adjusted_sound = sound.speedup(1.3, chunk_size=150, crossfade=73)

        # Play the adjusted audio
        play(adjusted_sound)
        TTS_Audio_play_queue.task_done()


# Queue for sentences
TTS_Audio_play_queue = queue.Queue()

# Start the worker thread
worker_thread = threading.Thread(
    target=process_TTS_Audio_play_queue, args=(TTS_Audio_play_queue,), daemon=True
)
worker_thread.start()


# Function to convert text to speech using edge-tts and play using pydub with speed adjustment
async def text_to_speech(text, speed=1, volume=1, voice="en-GB-MiaNeural"):
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


"""
 ######  ##     ##    ###    ######## 
##    ## ##     ##   ## ##      ##    
##       ##     ##  ##   ##     ##    
##       ######### ##     ##    ##    
##       ##     ## #########    ##    
##    ## ##     ## ##     ##    ##    
 ######  ##     ## ##     ##    ##    
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


# Define the function to split sentences
def split_sentence(response):
    delimiters = r"[.,;!?]"  # Add more delimiters if needed
    sentences = re.split(delimiters, response, maxsplit=1)
    if len(sentences) > 1:
        sentence, response = sentences[0], sentences[1]
    else:
        sentence, response = sentences[0], ""
    return sentence, response


# Function to process the queue
def process_TTS_queue(TTS_queue):
    while True:
        sentence = TTS_queue.get()
        if sentence is None:  # Sentinel value to stop the worker
            break
        asyncio.run(text_to_speech(sentence, speed=1))
        TTS_queue.task_done()


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
    relevant_context = ""
    # Connect to Milvus
    connections.connect(host="localhost", port="19530")
    collection = Collection(name="html_chunks")

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
        param={"metric_type": "L2", "params": {"nprobe": 10}},
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


def main():
    parser = argparse.ArgumentParser(description="Ollama Chat")
    parser.add_argument(
        "--model", default="phi3", help="Ollama model to use phi3 (default: llama3)"
    )
    args = parser.parse_args()

    # Example conversation history
    conversation_history = [{"role": "system", "content": "Welcome to Ollama Chat!"}]

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

        system_message = "You are a helpful assistant. You will give precise and concise answers from the given context"

        # Interact with the Ollama model
        if user_input:
            response = ollama_chat(
                user_input,
                system_message,
                args.model,
                conversation_history,
            )
            print(response)


if __name__ == "__main__":
    main()
