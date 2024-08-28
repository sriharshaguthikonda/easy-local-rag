import torch
import ollama
import os
import json
import argparse
from gtts import gTTS
from openai import OpenAI
import pygame
import io
import pynvml
import time
from tqdm import tqdm


import ChatTTS
import torchaudio


def play_wav_file(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    # Wait until music has finished playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.quit()


"""
TODO: first you will have to start the ollama server by using the command 
ollama serve
then you can run this script

TODO: TTS issue is solved with google tts as off now..but we will ahve to use a better tts solution.!

TODO: embddings saving is overwriting the old embeddings file.

"""

# Define temperature threshold in Celsius
TEMP_THRESHOLD = 51


# ANSI escape codes for colors
PINK = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# Constants
EMBEDDINGS_DIR = "Embeddings"
MOD_TIME_FILE = os.path.join(EMBEDDINGS_DIR, "mod_times.json")


# Function to read file content
def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# Function to save embeddings and file modification time
def save_embeddings(new_embeddings, mod_time, vault_name):
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    embeddings_file = os.path.join(EMBEDDINGS_DIR, f"{vault_name}_embeddings.pt")
    # Load existing embeddings if they exist
    if os.path.exists(embeddings_file):
        existing_embeddings = torch.load(embeddings_file)
        updated_embeddings = torch.cat([existing_embeddings, new_embeddings])
    else:
        updated_embeddings = new_embeddings
    # Save the updated embeddings
    torch.save(updated_embeddings, embeddings_file)
    with open(MOD_TIME_FILE, "w") as f:
        json.dump({vault_name: mod_time}, f)


# Function to load embeddings and file modification time
def load_embeddings(vault_name):
    embeddings_file = os.path.join(EMBEDDINGS_DIR, f"{vault_name}_embeddings.pt")
    if os.path.exists(embeddings_file) and os.path.exists(MOD_TIME_FILE):
        embeddings = torch.load(embeddings_file)
        with open(MOD_TIME_FILE, "r") as f:
            mod_time_data = json.load(f)
        mod_time = mod_time_data.get(vault_name)
        return embeddings, mod_time
    return None, None


# Function to check GPU temperature
def check_gpu_temperature():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    pynvml.nvmlShutdown()
    return temp


# Function to interact with the Ollama model
def ollama_chat(
    user_input,
    system_message,
    vault_embeddings,
    vault_content,
    ollama_model,
    conversation_history,
):
    # Configuration for the Ollama API client
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    # Get relevant context from the vault
    relevant_context = get_relevant_context(
        user_input, vault_embeddings, vault_content, top_k=3
    )
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)

    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input

    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})

    # Create a message history including the system message and the conversation history
    messages = [{"role": "system", "content": system_message}, *conversation_history]

    # Send the completion request to the Ollama model
    response = client.chat.completions.create(model=ollama_model, messages=messages)

    # Append the model's response to the conversation history
    conversation_history.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )

    # Return the content of the response from the model
    return response.choices[0].message.content


# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(
        model="mxbai-embed-large", prompt=rewritten_input
    )["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(
        torch.tensor(input_embedding).unsqueeze(0), vault_embeddings
    )
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context


# Function to generate embeddings for vault content with checkpointing and logging
def generate_embeddings(vault_content, vault_name, start_idx=0):
    vault_embeddings = []
    progress_log = os.path.join(EMBEDDINGS_DIR, f"{vault_name}_progress.json")

    # Load progress log if exists
    if os.path.exists(progress_log):
        with open(progress_log, "r") as f:
            progress_data = json.load(f)
        start_idx = progress_data.get("last_processed_idx", 0)
        print(f"Resuming from index {start_idx}")
    else:
        start_idx = 0

    for idx, content in enumerate(tqdm(vault_content, desc="Generating embeddings")):
        if idx < start_idx and idx != len(vault_content):
            continue

        # Check GPU temperature before processing each content
        while check_gpu_temperature() > TEMP_THRESHOLD:
            print("GPU temperature is too high. Pausing until temperature drops...")
            time.sleep(30)  # Sleep for 60 seconds before rechecking

        response = ollama.embeddings(model="mxbai-embed-large", prompt=content)
        if "embedding" in response:
            vault_embeddings.append(response["embedding"])
        else:
            print(f"Failed to get embedding for content: {content}")

        # Save checkpoint every 10 embeddings
        if (idx + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                EMBEDDINGS_DIR, f"{vault_name}_embeddings.pt"
            )
            # Check file modification time
            current_mod_time = os.path.getmtime("vault.txt")
            save_embeddings(
                torch.tensor(vault_embeddings), current_mod_time, vault_name
            )
            vault_embeddings = []
            with open(progress_log, "w") as f:
                json.dump({"last_processed_idx": idx}, f)

        # Save checkpoint every 10 embeddings
        if idx == len(vault_content):
            checkpoint_path = os.path.join(
                EMBEDDINGS_DIR, f"{vault_name}_embeddings.pt"
            )
            # Check file modification time
            current_mod_time = os.path.getmtime("vault.txt")
            save_embeddings(
                torch.tensor(vault_embeddings), current_mod_time, vault_name
            )
            vault_embeddings = []
            with open(progress_log, "w") as f:
                json.dump({"last_processed_idx": len(vault_content)}, f)

    return vault_embeddings


# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Ollama Chat")
    parser.add_argument(
        "--model", default="phi3", help="Ollama model to use (default: llama3)"
    )
    args = parser.parse_args()

    vault_files = [
        f for f in os.listdir() if f.startswith("vault") and f.endswith(".txt")
    ]

    for vault_file in vault_files:
        vault_name = os.path.splitext(vault_file)[0]
        vault_content = open_file(vault_file).splitlines()

        # Check file modification time
        current_mod_time = os.path.getmtime(vault_file)
        saved_embeddings, saved_mod_time = load_embeddings(vault_name)
        """

        TODO: this comparison of lenght of vault_content and first dimension of the saved_embeddings is checking if all the lines in the vault.txt file has a corrosponding embedding in the embeddings file.!

        """
        if os.path.exists(
            r"C:\Users\deletable\OneDrive\easy-local-rag\Embeddings\vault_embeddings.pt"
        ) and len(vault_content) == saved_embeddings.size(0):
            print(f"Loaded saved embeddings for {vault_name}")
            vault_embeddings = saved_embeddings
        else:
            print(f"Generating new embeddings for {vault_name}")
            vault_embeddings = generate_embeddings(vault_content, vault_name)
            vault_embeddings_tensor = torch.tensor(vault_embeddings)

        # Ensure the embeddings are in tensor format
        if not isinstance(vault_embeddings, torch.Tensor):
            vault_embeddings_tensor = torch.tensor(vault_embeddings)
        else:
            vault_embeddings_tensor = vault_embeddings

        # Initialize conversation history
        conversation_history = []

        while True:
            user_input = input(
                NEON_GREEN + "Enter your message (or 'exit' to quit): " + RESET_COLOR
            )
            if user_input.lower() == "exit":
                break

            system_message = "You are a helpful assistant. you will give concise answers from the given context and question"

            # Interact with the Ollama model
            if user_input:
                response_content = ollama_chat(
                    user_input,
                    system_message,
                    vault_embeddings_tensor,
                    vault_content,
                    args.model,
                    conversation_history,
                )

                # Print the response
                print(NEON_GREEN + response_content + RESET_COLOR)

            # Convert response to speech and play it

            # chat = ChatTTS.Chat()
    #        chat.load(compile=False)  # Set to True for better performance

    # wavs = chat.infer(response_content)

    # torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)

    # Example usage
    # play_wav_file("output1.wav")

    # Clean up NVML
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()


"""
TODO: first you will have to start the ollama server by using the command 
ollama serve
then you can run this script

TODO: TTS issue is solved with google tts as off now..but we will ahve to use a better tts solution.!

TODO: embddings saving is overwriting the old embeddings file.

"""
