import argparse
import sys

import ollama

from embedding_contract import EmbeddingContractError, validate_embedding_compatibility
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
import json


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

collection_name = "html_chunks_temp"


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

# Start the audio worker only after an accepted answer needs speech.
worker_thread = None


def _ensure_tts_audio_worker():
    global worker_thread
    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = threading.Thread(
            target=process_TTS_Audio_play_queue,
            args=(TTS_Audio_play_queue,),
            daemon=True,
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


def _router_for(provider, router=None, router_factory=None):
    if router is not None:
        return router
    if router_factory is not None:
        try:
            return router_factory(provider)
        except TypeError:
            return router_factory()
    from generation_models import groq_router, ollama_router

    return groq_router() if provider == "groq" else ollama_router()


_rewrite_router = None


def _router_for_rewrite(router=None):
    """Reuse one standalone Groq rewrite router for this process/session."""
    global _rewrite_router
    if router is not None:
        return router
    if _rewrite_router is None:
        _rewrite_router = _router_for("groq")
    return _rewrite_router


def _result_value(result, name, default=None):
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _retrieval_for_generation(
    user_input, provider, router=None, *, strict=False, cancelled=None
):
    options = {"top_k": 5, "rewrite": provider != "ollama"}
    if router is not None:
        options["router"] = router
    if strict:
        options["strict"] = True
    if cancelled is not None:
        options["cancelled"] = cancelled
    try:
        return get_relevant_context_hybrid(user_input, **options)
    except TypeError as error:
        if "rewrite" not in str(error):
            raise
        options.pop("rewrite", None)
        # Keep the selected session router and strict policy.  Dropping the
        # router here silently recreated the process-global Groq rewrite
        # client, bypassing the account/model policy used for the answer.
        return get_relevant_context_hybrid(user_input, **options)


def _accepted_speech(text):
    if not text or globals().get("dont_read_tts", False):
        return
    _ensure_tts_audio_worker()
    speech_queue = queue.Queue()
    speech_worker = threading.Thread(
        target=process_TTS_queue, args=(speech_queue,), daemon=True
    )
    speech_worker.start()
    speech_queue.put(text)
    speech_queue.put(None)


def _generate_chat_result(
    user_input,
    system_message,
    model_name,
    conversation_history,
    *,
    provider,
    router=None,
    strict=False,
    cancelled=None,
    on_chunk=None,
    **provider_options,
):
    retrieved = _retrieval_for_generation(
        user_input, provider, router=router, strict=strict, cancelled=cancelled
    )
    relevant_context = retrieved[0] if isinstance(retrieved, tuple) else (retrieved or "")
    user_content = (
        f"{relevant_context}\n\n{user_input}" if relevant_context else user_input
    )
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history,
        {"role": "user", "content": user_content},
    ]
    emitted = []

    def emit(chunk):
        if not chunk:
            return
        chunk = str(chunk)
        emitted.append(chunk)
        if on_chunk is None:
            print(NEON_GREEN + chunk, end="", flush=True)
        if on_chunk is not None:
            on_chunk(chunk)

    result = _router_for(provider, router=router).generate(
        preferred=model_name,
        messages=messages,
        strict=strict,
        stream=True,
        max_output_tokens=provider_options.pop("max_output_tokens", 1024),
        cancelled=cancelled,
        on_chunk=emit,
        **provider_options,
    )
    text = _result_value(result, "text", "") or "".join(emitted)
    try:
        if not _result_value(result, "text", None):
            result.text = text
    except Exception:
        pass
    conversation_history.append({"role": "user", "content": user_content})
    conversation_history.append({"role": "assistant", "content": text})
    _accepted_speech(text)
    if text and on_chunk is None:
        print(RESET_COLOR + "\n")
    return result


def generate_answer(
    user_input,
    system_message,
    model_name,
    conversation_history,
    *,
    provider="groq",
    router=None,
    strict=False,
    cancelled=None,
    on_chunk=None,
    **provider_options,
):
    return _generate_chat_result(
        user_input,
        system_message,
        model_name,
        conversation_history,
        provider=provider,
        router=router,
        strict=strict,
        cancelled=cancelled,
        on_chunk=on_chunk,
        **provider_options,
    )

def chat_with_model(
    user_input,
    system_message,
    groq_model,
    ollama_model,
    conversation_history,
    *,
    provider="groq",
    router=None,
    strict=False,
    cancelled=None,
    on_chunk=None,
    return_result=False,
    **provider_options,
):
    model_name = groq_model if provider == "groq" else ollama_model
    result = generate_answer(
        user_input,
        system_message,
        model_name,
        conversation_history,
        provider=provider,
        router=router,
        strict=strict,
        cancelled=cancelled,
        on_chunk=on_chunk,
        **provider_options,
    )
    return result if return_result else _result_value(result, "text", "")


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
    *,
    router=None,
    strict=False,
    cancelled=None,
    on_chunk=None,
    **provider_options,
):
    return _result_value(
        generate_answer(
            user_input,
            system_message,
            ollama_model,
            conversation_history,
            provider="ollama",
            router=router,
            strict=strict,
            cancelled=cancelled,
            on_chunk=on_chunk,
            **provider_options,
        ),
        "text",
        "",
    )


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

# Provider clients are created by generation_models after an explicit mode
# selection.  Keep the name for compatibility with legacy callers.
client = None


def groq_chat(
    user_input,
    system_message,
    groq_model,
    conversation_history,
    *,
    router=None,
    strict=False,
    cancelled=None,
    on_chunk=None,
    **provider_options,
):
    return _result_value(
        generate_answer(
            user_input,
            system_message,
            groq_model,
            conversation_history,
            provider="groq",
            router=router,
            strict=strict,
            cancelled=cancelled,
            on_chunk=on_chunk,
            **provider_options,
        ),
        "text",
        "",
    )


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


def rewrite_input_and_generate_synonyms(
    original_input, *, router=None, strict=False, cancelled=None
):
    """Rewrite search text through the shared Groq boundary."""
    system_prompt = (
        "You are a helpful assistant working in a medical context. Your tasks are:\n"
        "1) Rephrase the given input to make it clearer and more precise in one sentence while preserving its original meaning. It will be used for searching ChromaDB after conversion to embeddings.\n"
        "2) Provide a list of synonyms for each keyword in the rephrased input.\n"
        "3) Provide spelling variants (if applicable, such as American and British spellings) for each keyword.\n"
        "4) Provide plural and singular forms for each keyword.\n"
        "5) Provide other parts of speech forms for each keyword.\n"
        "6) Provide closely related terms for each keyword.\n"
        "Respond in the following JSON format:\n"
        "{\n"
        '  "rephrased": "[sentence]",\n'
        '  "keywords": {\n'
        '    "[word]": {\n'
        '      "synonyms": ["synonym1", "synonym2"],\n'
        '      "spelling_variants": ["variant1", "variant2"],\n'
        '      "plural_singular": ["plural_form", "singular_form"],\n'
        '      "parts_of_speech": ["noun form", "verb form"],\n'
        '      "related_terms": ["related1", "related2"]\n'
        "    }\n"
        "  }\n"
        "}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Rewrite and generate synonyms, spelling variants, "
                "plural/singular forms, parts of speech, and related terms "
                f'for: "{original_input}".'
            ),
        },
    ]
    try:
        result = _router_for_rewrite(router).generate(
            preferred=groq_model,
            messages=messages,
            strict=strict,
            cancelled=cancelled,
            stream=False,
            max_output_tokens=512,
            response_format={"type": "json_object"},
        )
        raw = _result_value(result, "text", "") or ""
        if isinstance(raw, dict):
            data = raw
        else:
            raw = str(raw).strip()
            if raw.startswith(chr(96) * 3):
                raw = raw.strip(chr(96))
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            data = json.loads(raw)
        rewritten = data.get("rephrased") or data.get("rewritten") or original_input
        keywords = data.get("keywords") or {}
        return rewritten, keywords if isinstance(keywords, dict) else {}
    except Exception as error:
        print(f"Rewrite unavailable; using original query ({type(error).__name__}).")
        return original_input, {}


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
    *,
    rewrite=True,
    router=None,
    strict=False,
    cancelled=None,
):
    try:
        relevant_context = ""
        if rewrite:
            rewrite_options = {}
            if router is not None:
                rewrite_options["router"] = router
            if strict:
                rewrite_options["strict"] = True
            if cancelled is not None:
                rewrite_options["cancelled"] = cancelled
            rewritten_input, synonym_dict = rewrite_input_and_generate_synonyms(
                user_input, **rewrite_options
            )
        else:
            rewritten_input, synonym_dict = user_input, {}

        if callable(cancelled) and cancelled():
            return "", []

        local_collection = collection

        requested_model = model
        if not validate_embedding_compatibility(local_collection, requested_model):
            return "", []

        # Encode the rewritten input into an embedding
        input_embedding = ollama.embeddings(
            model=requested_model,
            prompt=rewritten_input,
            keep_alive=-1,
        )["embedding"]

        # Perform vector similarity search
        search_result = local_collection.query(
            query_embeddings=[input_embedding],
            n_results=50,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        # Extract results with distances
        vector_results = [
            {
                "document": document,
                "meta": meta,
                "embedding": emb,
                "vector_score": 1.0
                - dist,  # Convert distance to similarity (assuming normalized)
            }
            for document, meta, dist, emb in zip(
                search_result["documents"][0],
                search_result["metadatas"][0],
                search_result["distances"][0],
                search_result["embeddings"][0],
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

            for document, meta, emb in zip(
                search_result["documents"][0],
                search_result["metadatas"][0],
                search_result["embeddings"][0],
            ):
                # Match against both original keywords and their synonyms
                match_score = sum(
                    document.lower().count(keyword)
                    + meta["file_name"].lower().count(keyword)
                    for keyword in keywords
                )
                if match_score > 0:
                    keyword_results.append(
                        {
                            "document": document,
                            "meta": meta,
                            "embedding": emb,
                            "keyword_score": match_score,
                        }
                    )

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
                "document": res["document"],
                "meta": res["meta"],
                "embedding": res["embedding"],
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
                    "document": res["document"],
                    "meta": res["meta"],
                    "embedding": res.get("embedding"),
                    "final_score": beta * res["keyword_score"],
                }

        # Sort by final_score
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["final_score"], reverse=True
        )

        # Seed MMR with the top-k results, then add one diverse result at a time.
        selected_results = sorted_results[:top_k]
        remaining_results = [res for res in sorted_results[top_k:]]
        for _ in range(additional_unique_files):
            if not remaining_results:
                break

            best_result = None
            best_score = None
            for res in remaining_results:
                current_embedding = res.get("embedding")
                max_similarity = max(
                    [
                        np.dot(current_embedding, selected["embedding"])
                        for selected in selected_results
                        if current_embedding is not None
                        and selected.get("embedding") is not None
                    ],
                    default=0,
                )
                mmr_score = (
                    res["final_score"]
                    if current_embedding is None
                    else lambda_mmr * res["final_score"]
                    - (1 - lambda_mmr) * max_similarity
                )
                if best_score is None or mmr_score > best_score:
                    best_result = res
                    best_score = mmr_score

            selected_results.append(best_result)
            remaining_results.remove(best_result)

        final_results = [
            {
                **res["meta"],
                "document": res["document"],
                "score": res["final_score"],
            }
            for res in selected_results
        ]

        # Prepare relevant context
        relevant_context = "\n\n".join(
            source["document"] for source in final_results
        )

        # Start a worker thread to print details of the results
        worker_thread = threading.Thread(
            target=print_relevant_context,
            args=(selected_results,),
            daemon=True,
        )
        worker_thread.start()

        # Keep text and metadata together so callers cannot lose retrieved evidence.
        return relevant_context, final_results

    except Exception as e:
        if isinstance(e, EmbeddingContractError):
            raise RuntimeError(f"Retrieval failed: {e}") from e
        raise RuntimeError("Retrieval failed; no model request was made") from e


def print_relevant_context(results):
    print("Context Pulled from Documents:\n")
    for result in results:
        meta = result["meta"]
        file_name = meta["file_name"]
        modification_time = meta.get("modification_time", "Unknown")
        text = result["document"]

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
        settings=Settings(anonymized_telemetry=False),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    collection = client.get_collection(collection_name)
    return collection


# Modify main() to use initialize_collection
def build_parser():
    parser = argparse.ArgumentParser(description="Easy local RAG Streamlit backend")
    parser.add_argument("--provider", choices=("groq", "ollama"), default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="preferred provider model; bare --model retains Ollama meaning",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--refresh-models", action="store_true")
    parser.add_argument("--strict-model", action="store_true")
    parser.add_argument("--offline-search", metavar="QUERY")
    return parser


def _open_backend_collection(path=None, name=collection_name):
    if path is None:
        return initialize_collection()
    local_client = chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    return local_client.get_collection(name)


def _safe_error_code(error, default="generation_error"):
    code = getattr(error, "code", None)
    if not code:
        return default
    return re.sub(r"[^a-z0-9_.-]+", "_", str(code).lower()).strip("_") or default


def _print_model_catalogue(router, stdout, refresh=False, provider=None):
    provider_label = provider or "selected provider"
    try:
        profiles = router.catalogue(refresh=refresh)
    except Exception as error:
        stdout.write(
            f"Model catalogue unavailable for {provider_label}: "
            f"{_safe_error_code(error, 'catalogue_unavailable')}\n"
        )
        return None
    if not profiles:
        stdout.write(f"No models available for {provider_label}.\n")
        return []
    for profile in profiles:
        model_id = _result_value(profile, "id", _result_value(profile, "model_id", ""))
        capabilities = _result_value(profile, "capabilities", ())
        stdout.write(f"{model_id}\t{sorted(capabilities) if capabilities else ''}\n")
    return profiles


def _write_offline_results(local_collection, query, searcher, stdout):
    try:
        results = searcher(local_collection, query, limit=5)
        stdout.write(json.dumps(results, default=str) + "\n")
        return True
    except Exception as error:
        stdout.write(f"Offline search failed: {_safe_error_code(error, 'offline_search_failed')}\n")
        return False

def main(
    argv=None,
    *,
    router_factory=None,
    collection_opener=None,
    searcher=None,
    input_fn=input,
    stdout=None,
):
    global collection, conversation_history, dont_read_tts, just_query_file_search
    stdout = stdout or sys.stdout
    args = build_parser().parse_args(argv)

    if args.offline_search is not None:
        if collection_opener is None:
            from offline_search import open_local_collection

            collection_opener = open_local_collection
        if searcher is None:
            from offline_search import search_documents

            searcher = search_documents
        local_collection = collection_opener(
            "./chroma", collection_name
        )
        return 0 if _write_offline_results(
            local_collection, args.offline_search, searcher, stdout
        ) else 1

    provider = args.provider or ("ollama" if args.model else "groq")
    model_name = args.model or (groq_model if provider == "groq" else ollama_model)
    router = None

    if args.list_models or args.refresh_models:
        try:
            router = _router_for(provider, router_factory=router_factory)
        except Exception as error:
            stdout.write(
                f"Model catalogue unavailable for {provider}: "
                f"{_safe_error_code(error, 'catalogue_unavailable')}\n"
            )
            return 1
        profiles = _print_model_catalogue(
            router, stdout, refresh=args.refresh_models, provider=provider
        )
        if profiles is None:
            return 1
        if args.list_models:
            return 0

    if collection_opener is None:
        collection_opener = lambda _path, _name: _open_backend_collection()
    collection = collection_opener(None, collection_name)
    conversation_history = [
        {"role": "system", "content": "Welcome to Easy Local RAG."}
    ]
    dont_read_tts = False
    just_query_file_search = False
    if router is None:
        try:
            router = _router_for(provider, router_factory=router_factory)
        except Exception as error:
            stdout.write(f"Generation failed: {_safe_error_code(error)}\n")
            return 1

    system_message = (
        "You are a helpful assistant. Give precise, concise answers from the "
        "provided context."
    )
    while True:
        try:
            user_input = input_fn("Enter your message (or 'exit' to quit): ")
        except (EOFError, KeyboardInterrupt):
            stdout.write("\n")
            break
        if user_input is None or user_input.strip().lower() in {"exit", "quit"}:
            break
        stripped_input = re.sub(r"s{4,}", "", user_input, flags=re.IGNORECASE)
        stripped_input = re.sub(r"q{4,}", "", stripped_input, flags=re.IGNORECASE).strip()
        lowered_input = user_input.lower()
        dont_read_tts = bool(re.search(r"s{4,}", lowered_input))
        just_query_file_search = bool(re.search(r"q{4,}", lowered_input))
        if not stripped_input:
            continue

        if just_query_file_search:
            if searcher is None:
                from offline_search import search_documents

                searcher = search_documents
            _write_offline_results(collection, stripped_input, searcher, stdout)
            continue

        try:
            streamed = []

            def emit_chunk(chunk):
                streamed.append(str(chunk))
                stdout.write(str(chunk))
                stdout.flush()

            result = generate_answer(
                stripped_input,
                system_message,
                model_name,
                conversation_history,
                provider=provider,
                router=router,
                strict=args.strict_model,
                on_chunk=emit_chunk,
            )
            stdout.write(
                "provider="
                f"{provider} "
                "preferred_model="
                f"{_result_value(result, 'preferred_model', model_name)} "
                "actual_model="
                f"{_result_value(result, 'actual_model', model_name)} "
                "reason="
                f"{_result_value(result, 'reason', '') or 'none'}\n"
            )
            if streamed:
                stdout.write("\n")
            else:
                stdout.write(f"{_result_value(result, 'text', '')}\n")
        except KeyboardInterrupt:
            stdout.write("\nGeneration cancelled.\n")
            break
        except Exception as error:
            stdout.write(f"Generation failed: {_safe_error_code(error)}\n")
    return 0

        # print("stripped_input :", stripped_input)


if __name__ == "__main__":
    main()
