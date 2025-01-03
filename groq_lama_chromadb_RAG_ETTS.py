import argparse

import ollama
from groq import Groq

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

# import edge_tts
from gtts import gTTS

import asyncio
import nest_asyncio

from dotenv import load_dotenv


import numpy as np


# Constants
EMBEDDINGS_DIR = "Embeddings"
model = "mxbai-embed-large"
# groq_model="llama3-70b-8192"
# groq_model = "llama-3.1-70b-versatile"
groq_model = "llama-3.3-70b-versatile"
ollama_model = "phi-3"

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


# additional_unique_files
# varriable is int the get_relevant_context/get_relevant_context_hybrid function


non_keywords = set(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "for",
        "to",
        "from",
        "up",
        "down",
        "into",
        "over",
        "under",
        "about",
        "between",
        "after",
        "before",
        "while",
        "during",
        "as",
        "but",
        "or",
        "so",
        "such",
        "that",
        "this",
        "these",
        "those",
        "all",
        "any",
        "both",
        "some",
        "most",
        "much",
        "many",
        "few",
        "one",
        "each",
        "every",
        "neither",
        "either",
        "who",
        "whom",
        "whose",
        "what",
        "which",
        "where",
        "when",
        "why",
        "how",
        "it",
        "its",
        "itself",
        "he",
        "she",
        "her",
        "him",
        "his",
        "they",
        "their",
        "them",
        "themselves",
        "you",
        "your",
        "yourself",
        "yourselves",
        "we",
        "our",
        "ours",
        "us",
    ]
)

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


# Updated queue processing function
def process_TTS_Audio_play_queue(TTS_Audio_play_queue):
    while True:
        try:
            audio_fp = TTS_Audio_play_queue.get()
            if audio_fp is None:
                break  # Exit loop if sentinel is encountered
            audio_fp.seek(0)  # Reset pointer
            audio = AudioSegment.from_file(audio_fp, format="mp3")
            play(audio)  # Play the audio
            TTS_Audio_play_queue.task_done()
        except Exception as e:
            print(f"Error occurred in queue processing: {e}")


# Queue for sentences
TTS_Audio_play_queue = queue.Queue()

# Start the worker thread
worker_thread = threading.Thread(
    target=process_TTS_Audio_play_queue, args=(TTS_Audio_play_queue,), daemon=True
)
worker_thread.start()


# Function to convert text to speech using edge-tts and play using pydub with speed adjustment
async def text_to_speech_gtts(text, speed=1.2, volume=1.0, lang="en", tld="co.uk"):
    try:
        if not text.strip():
            raise ValueError("Text is empty. Cannot synthesize speech.")

        loop = asyncio.get_event_loop()
        audio_fp = await loop.run_in_executor(
            None, generate_gtts_audio, text, lang, tld
        )

        audio = AudioSegment.from_file(audio_fp, format="mp3")
        audio = audio.speedup(playback_speed=speed)
        audio = audio + (volume * 10)

        processed_audio_fp = io.BytesIO()
        audio.export(processed_audio_fp, format="mp3")
        processed_audio_fp.seek(0)

        TTS_Audio_play_queue.put(processed_audio_fp)

    except Exception as e:
        print(f"Error during async TTS processing: {e}")


def generate_gtts_audio(text, lang, tld):
    tts = gTTS(text=text, lang=lang, tld=tld)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp


# Function to process the queue
def process_TTS_queue(TTS_queue):
    global dont_read_tts
    while True:
        sentence = TTS_queue.get()
        if sentence is None:  # Sentinel value to stop the worker
            break
        if dont_read_tts:
            dont_read_tts = False  # Reset the flag after skipping
        else:
            asyncio.run(text_to_speech_gtts(sentence, speed=1.4))
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
    global just_query_file_search
    # Get relevant context from Milvus
    relevant_context = get_relevant_context_hybrid(user_input, top_k=5)

    # Prepare the user's input by concatenating it with the relevant context
    if relevant_context:
        user_input_with_context = relevant_context + "\n\n" + user_input
    else:
        user_input_with_context = user_input

    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})

    # Create a message history including the system message and the conversation history
    messages = [{"role": "system", "content": system_message}, *conversation_history]

    if just_query_file_search is False:
        just_query_file_search = True

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

    else:
        pass

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
    global just_query_file_search
    response = ""

    # Get relevant context from Milvus
    relevant_context = get_relevant_context_hybrid(user_input, top_k=5)

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

    # Step 1: Calculate total tokens and adjust dynamically
    max_tpm = 6000  # Token per minute limit
    input_tokens = count_tokens(messages, groq_model)

    # Ensure total tokens (input + response) stay within the TPM limit
    available_tokens_for_response = max(max_tpm - input_tokens, 0)
    max_response_tokens = min(6000, available_tokens_for_response)

    if input_tokens > max_tpm:
        # Step 2: Trim conversation history and user input
        while input_tokens > max_tpm:
            if conversation_history:
                # Remove the oldest non-system message from conversation history
                for i, msg in enumerate(conversation_history):
                    if msg["role"] != "system":
                        del conversation_history[i]
                        break
            else:
                # Truncate user input if no more history can be removed
                user_input = user_input[
                    : len(user_input) - 20
                ]  # Remove in chunks of 20 characters
                messages[-1] = {"role": "user", "content": user_input}

            # Recalculate total tokens
            messages = [
                {"role": "system", "content": system_message},
                *conversation_history,
                {"role": "user", "content": user_input},
            ]
            input_tokens = count_tokens(messages, groq_model)

        # Update available tokens for response after adjustments
        available_tokens_for_response = max(max_tpm - input_tokens, 0)
        max_response_tokens = min(6000, available_tokens_for_response)

    # Step 3: Send the request
    if just_query_file_search is False:
        just_query_file_search = True

        stream = client.chat.completions.create(
            # Required parameters
            messages=messages,
            model=groq_model,
            temperature=1,
            max_tokens=max_response_tokens,  # Dynamically adjusted
            top_p=1,
            stop="",
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

            if any(delimiter in response for delimiter in ".:!?"):
                response = response[1:]  # Remove the first character
                sentence, response = split_sentence(response)
                TTS_queue.put(sentence)

        # Print the response
        print(RESET_COLOR + "\n")

    # Append the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response})

    # Step 4: Trim conversation history for the next cycle if needed
    while count_tokens(conversation_history, groq_model) > max_tpm:
        for i, msg in enumerate(conversation_history):
            if msg["role"] != "system":
                del conversation_history[i]
                break

    return response


def count_tokens(messages, model="llama-3.3-70b-versatile"):
    # Replace this with your token counting logic (or simple tokenizer if tiktoken is unavailable)
    def simple_tokenizer(text):
        import re

        return re.findall(r"\w+|[^\s\w]", text)

    total_tokens = 0
    for message in messages:
        total_tokens += len(simple_tokenizer(message["content"]))
    return total_tokens


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
########  ######## ##      ## ########  #### ######## ######## 
##     ## ##       ##  ##  ## ##     ##  ##     ##    ##       
##     ## ##       ##  ##  ## ##     ##  ##     ##    ##       
########  ######   ##  ##  ## ########   ##     ##    ######   
##   ##   ##       ##  ##  ## ##   ##    ##     ##    ##       
##    ##  ##       ##  ##  ## ##    ##   ##     ##    ##       
##     ## ########  ###  ###  ##     ## ####    ##    ######## 
"""


def rewrite_input_and_generate_synonyms(original_input):
    try:
        # Single API call to get rewritten input, synonyms, and spelling variants
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Your task is threefold: "
                        "1) Rewrite the given input to make it clearer and more precise in one sentence while preserving its original meaning. "
                        "2) Provide a list of synonyms for each keyword in the rewritten input. "
                        "3) Provide spelling variants (if applicable, such as American and British spellings) for each keyword. "
                        "Respond with the rewritten sentence followed by the synonyms and spelling variants in the format: "
                        "'Rewritten: [sentence]' and 'Keywords: [word1: synonym1, synonym2; spelling variant1, spelling variant2; "
                        "word2: synonym1, synonym2; spelling variant1, spelling variant2]'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Rewrite and generate synonyms and spelling variants for: "{original_input}".""",
                },
            ],
            model=groq_model,
            temperature=0.7,
            stream=False,
        )

        # Parse the response
        response_text = chat_completion.choices[0].message.content.strip()
        print(f"Full Response:\n{response_text}")

        # Extract rewritten sentence and keywords
        rewritten_input = None
        synonym_and_variant_dict = {}

        if "Rewritten:" in response_text and "Keywords:" in response_text:
            parts = response_text.split("Keywords:")
            rewritten_input = parts[0].replace("Rewritten:", "").strip()
            keywords_raw = parts[1].strip()

            # Parse the keywords into a dictionary
            for entry in keywords_raw.split(";"):
                if ":" in entry:
                    try:
                        keyword, details = entry.split(":", 1)
                        keyword = keyword.strip()
                        details_split = [d.strip() for d in details.split(";")]

                        # Separate synonyms and variants
                        synonyms = [
                            d
                            for d in details_split
                            if not d.startswith("no variants")
                            and not d.startswith("spelling variant")
                        ]
                        spelling_variants = [
                            d.replace("spelling variant", "").strip()
                            for d in details_split
                            if "spelling variant" in d
                        ]

                        # Handle "no synonyms available"
                        if "no synonyms available" in details_split:
                            synonyms = []

                        synonym_and_variant_dict[keyword] = {
                            "synonyms": synonyms,
                            "spelling_variants": spelling_variants,
                        }
                    except ValueError:
                        print(f"Skipping malformed entry: {entry}")

        return rewritten_input, synonym_and_variant_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, {}


"""
 ######   #######  ##    ## ######## ######## ##     ## ######## 
##    ## ##     ## ###   ##    ##    ##        ##   ##     ##    
##       ##     ## ####  ##    ##    ##         ## ##      ##    
##       ##     ## ## ## ##    ##    ######      ###       ##    
##       ##     ## ##  ####    ##    ##         ## ##      ##    
##    ## ##     ## ##   ###    ##    ##        ##   ##     ##    
 ######   #######  ##    ##    ##    ######## ##     ##    ##    
"""


def get_relevant_context_hybrid(
    user_input,
    top_k=5,
    additional_unique_files=5,
    keyword_match=True,
    alpha=0.7,  # Weight for vector similarity
    beta=0.3,  # Weight for keyword match
    lambda_mmr=0.5,  # Balance parameter for MMR
):
    try:
        relevant_context = ""
        rewritten_input, synonym_dict = rewrite_input_and_generate_synonyms(user_input)

        # Encode the rewritten input into an embedding
        input_embedding = ollama.embeddings(
            model=model,
            prompt=rewritten_input,
            keep_alive=-1,
        )["embedding"]

        # Perform vector similarity search
        search_result = collection.query(
            query_embeddings=[input_embedding],
            n_results=50,
            include=["documents", "metadatas", "distances"],
        )

        # Extract results with distances
        vector_results = [
            {
                "meta": meta,
                "vector_score": 1.0
                - dist,  # Convert distance to similarity (assuming normalized)
            }
            for meta, dist in zip(
                search_result["metadatas"][0], search_result["distances"][0]
            )
        ]

        # Perform keyword matching if enabled
        keyword_results = []
        if keyword_match:
            # Extract keywords and synonyms
            keywords = [
                word
                for word in rewritten_input.lower().split()
                if word not in non_keywords
            ]

            # Add synonyms to the keyword list
            for key, synonyms in synonym_dict.items():
                keywords.extend(synonyms)

            # Use a set to remove duplicates
            keywords = set(keywords)

            for meta in search_result["metadatas"][0]:
                # Match against both original keywords and their synonyms
                match_score = sum(
                    meta["text"].lower().count(keyword)
                    + meta["file_name"].lower().count(keyword)
                    for keyword in keywords
                )
                if match_score > 0:
                    keyword_results.append({"meta": meta, "keyword_score": match_score})

        # Normalize scores for both vector and keyword results
        max_vector_score = max(
            [res["vector_score"] for res in vector_results], default=1
        )
        max_keyword_score = max(
            [res["keyword_score"] for res in keyword_results], default=1
        )

        for res in vector_results:
            res["vector_score"] /= max_vector_score

        for res in keyword_results:
            res["keyword_score"] /= max_keyword_score

        # Combine results using weighted scoring
        combined_results = {}
        for res in vector_results:
            file_name = res["meta"]["file_name"]
            combined_results[file_name] = {
                "meta": res["meta"],
                "final_score": alpha * res["vector_score"],
            }

        for res in keyword_results:
            file_name = res["meta"]["file_name"]
            if file_name in combined_results:
                combined_results[file_name]["final_score"] += (
                    beta * res["keyword_score"]
                )
            else:
                combined_results[file_name] = {
                    "meta": res["meta"],
                    "final_score": beta * res["keyword_score"],
                }

        # Sort by final_score
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["final_score"], reverse=True
        )

        # Limit results to top_k
        final_results = [res["meta"] for res in sorted_results[:top_k]]

        # Use MMR to select additional_unique_files
        remaining_results = [res for res in sorted_results[top_k:]]
        selected_additional_files = []

        for res in remaining_results:
            max_similarity = max(
                [
                    np.dot(res["meta"]["embedding"], selected["meta"]["embedding"])
                    for selected in selected_additional_files
                ],
                default=0,
            )
            mmr_score = (
                lambda_mmr * res["final_score"] - (1 - lambda_mmr) * max_similarity
            )
            res["mmr_score"] = mmr_score

        # Sort by MMR score to get the most relevant and diverse results
        sorted_additional_files = sorted(
            remaining_results, key=lambda x: x["mmr_score"], reverse=True
        )

        selected_additional_files = sorted_additional_files[:additional_unique_files]

        # Combine the top_k and the selected additional unique files
        final_results.extend([res["meta"] for res in selected_additional_files])

        # Prepare relevant context
        relevant_context = "\n\n".join([res["text"] for res in final_results])

        # Start a worker thread to print details of the results
        worker_thread = threading.Thread(
            target=print_relevant_context,
            args=(final_results,),
            daemon=True,
        )
        worker_thread.start()

        return relevant_context

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Answer this yourself!"


def print_relevant_context(results):
    print("Context Pulled from Documents:\n")
    for meta in results:
        file_name = meta["file_name"]
        modification_time = meta.get("modification_time", "Unknown")
        text = meta["text"]

        clickable_file_path = urljoin("file:", Path(file_name).as_uri())

        print(
            f"{YELLOW}Text:{RESET_COLOR} {text}\n"
            f"{BLUE}File Name:{clickable_file_path}\n{RESET_COLOR}"
            f"{PINK}Modification Time: {modification_time}\n{RESET_COLOR}"
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

# Make collection a global variable
collection = None


def initialize_collection():
    global collection
    client = chromadb.PersistentClient(
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    collection = client.get_collection(collection_name)
    return collection


# Modify main() to use initialize_collection
def main():
    global collection, conversation_history, dont_read_tts, just_query_file_search
    # Reset conversation history

    system_message = "You are a helpful assistant. You will give precise and concise answers from the given context. if the context doesnot have the answer then give it from your knowledge"

    dont_read_tts = False
    just_query_file_search = False

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

    collection = initialize_collection()

    get_relevant_context_hybrid(
        user_input="just loading ollama embeddings model and chromadb, dont respond"
    )

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

        # Strip out the extra "ssss" and "qqqq" from the user input
        stripped_input = re.sub(r"s{4,}", "", user_input.lower())
        stripped_input = re.sub(r"q{4,}", "", stripped_input)
        stripped_input = stripped_input.strip()

        # Check and set flags
        dont_read_tts = bool(re.search(r"s{4,}", user_input.lower()))
        just_query_file_search = bool(re.search(r"q{4,}", user_input.lower()))

        # Interact with the Ollama model
        if stripped_input:
            chat_with_model(
                stripped_input,
                system_message,
                groq_model,
                ollama_model,
                conversation_history=conversation_history,
            )

        # print("stripped_input :", stripped_input)


# Initialize collection when module is imported
if collection is None:
    collection = initialize_collection()

if __name__ == "__main__":
    main()
