import subprocess


def check_and_prompt_start_ollama():
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
            subprocess.run(["ollama", "serve"], check=True)
            print("Ollama has been started.")
    except subprocess.CalledProcessError:
        subprocess.run(["ollama", "serve"], check=True)
        print("Ollama has been started.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Call the function
check_and_prompt_start_ollama()
