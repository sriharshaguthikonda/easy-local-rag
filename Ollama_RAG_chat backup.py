import argparse
import json
import os
import torch
import ollama
from tqdm import tqdm
from gtts import gTTS
import pygame
import io


"""
TODO: vault_embeddings.pt has been replaced wiht vault_embeddings.txt
"""
# Constants
EMBEDDINGS_DIR = "Embeddings"

# ANSI escape codes for colors
PINK = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# Magenta
MAGENTA = "\033[35m"
# Blue
BLUE = "\033[94m"
# Red
RED = "\033[91m"
# Bold
BOLD = "\033[1m"


# Function to convert text to speech using gTTS and play using pygame with speed adjustment
def text_to_speech(text, speed=1.0, volume=0.5):
    try:
        tts = gTTS(
            lang="en-gb",
            text=text,
            tld="co.uk",
        )
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)

        pygame.mixer.init()
        pygame.mixer.music.load(audio_fp)

        # Set the volume (range 0.0 to 1.0)
        pygame.mixer.music.set_volume(volume)

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.quit()
    except Exception as e:
        print(f"Error occurred during playback: {e}")


# Function to load vault content from JSON file
def load_vault_content(json_file):
    vault_content = {}
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            for chunk in entry["chunks"]:
                vault_content[chunk["id"]] = chunk["text"].strip()
    return vault_content


# Function to read file content
def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# Function to interact with the Ollama model
def ollama_chat(
    user_input,
    system_message,
    vault_embeddings,
    vault_content,
    ollama_model,
    conversation_history,
):
    # Get relevant context from the vault
    relevant_context = get_relevant_context(
        user_input, vault_embeddings, vault_content, top_k=3
    )
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n\n".join(relevant_context)
        print(
            "Context Pulled from Documents: \n\n"
            + CYAN
            + context_str
            + RESET_COLOR
            + "\n\n"
        )
    else:
        print("No relevant context found.")

    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input

    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})

    # Create a message history including the system message and the conversation history
    messages = [{"role": "system", "content": system_message}, *conversation_history]

    # Send the completion request to the Ollama model with stream=True
    stream = ollama.chat(
        model=ollama_model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        print(NEON_GREEN + chunk["message"]["content"], end="", flush=True)

    # Print the response
    print(RESET_COLOR + "\n")


# Function to load embeddings and file modification time
def load_embeddings():
    embeddings_file = os.path.join(EMBEDDINGS_DIR, "all-minilm_vault_embeddings.pt")
    if os.path.exists(embeddings_file):
        embeddings = torch.load(embeddings_file)
        return embeddings
    return []


# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if not vault_embeddings:  # Check if the list is empty
        return []

    # Encode the rewritten input
    # input_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=rewritten_input)["embedding"]
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model="all-minilm", prompt=rewritten_input)[
        "embedding"
    ]

    # Create a tensor from input_embedding
    input_embedding_tensor = torch.tensor(input_embedding).unsqueeze(0)

    # Prepare embeddings and ids lists
    embeddings_list = []
    ids_list = []

    for embedding_dict in vault_embeddings:
        for chunk_id, embedding in embedding_dict.items():
            embeddings_list.append(embedding)
            ids_list.append(chunk_id)

    # Create a tensor from the embeddings list
    vault_embeddings_tensor = torch.tensor(embeddings_list)

    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(
        input_embedding_tensor, vault_embeddings_tensor
    )

    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))

    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()

    # Get the corresponding chunk_ids from the top indices
    top_chunk_ids = [ids_list[idx] for idx in top_indices]

    # Get the corresponding context from the vault content using chunk_ids
    relevant_context = [vault_content[chunk_id] for chunk_id in top_chunk_ids]
    print(chunk_id)

    return relevant_context


def main():
    parser = argparse.ArgumentParser(description="Ollama Chat")
    parser.add_argument(
        "--model", default="phi3", help="Ollama model to use (default: llama3)"
    )
    args = parser.parse_args()

    # Example conversation history
    conversation_history = [{"role": "system", "content": "Welcome to Ollama Chat!"}]
    vault_embeddings = load_embeddings()

    # Example usage
    json_file = "vault.json"
    vault_content = load_vault_content(json_file)

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

        system_message = "You are a helpful assistant. You will give precise and concise answers from the given context and question"

        # Interact with the Ollama model
        if user_input:
            ollama_chat(
                user_input,
                system_message,
                vault_embeddings,
                vault_content,
                args.model,
                conversation_history,
            )


if __name__ == "__main__":
    main()
