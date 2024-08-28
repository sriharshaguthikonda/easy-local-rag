"""from gtts import gTTS
import io
from pydub import AudioSegment
from pydub.playback import play


# Function to convert text to speech using gTTS and play using pygame with speed adjustment
def text_to_speech(text, speed=1.3, volume=1):
    try:
        tts = gTTS(
            lang="en-gb",
            text=text,
            tld="co.uk",
            slow=False,
        )
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)

        sound = AudioSegment.from_file(audio_fp)

        # Adjust the speed (150ms chunks with 25ms crossfade)
        adjusted_sound = sound.speedup(speed, chunk_size=150, crossfade=73)

        # Play the adjusted audio
        play(adjusted_sound)

    except Exception as e:
        print(f"Error occurred during playback: {e}")


text_to_speech(
    "The dictum by Albert Szent-Györgyi regarding leadership in academic medicine states that the capabilities and conditions for organizational success in this field involve effective collaboration, innovation, and dedication to advancing scientific research. This concept aligns with his broader contributions to biology and chemistry, highlighting the importance of interdisciplinary approaches and continuous learning within medical institutions."
)
"""

"""
import pyttsx3

# Initialize pyttsx3 engine
engine = pyttsx3.init()


# Function to convert text to speech using pyttsx3 and play immediately
def text_to_speech(text, speed=1.0, volume=0.5):
    try:
        # Set speed and volume
        engine.setProperty("rate", speed * 150)  # Speed adjustment
        engine.setProperty("volume", volume)  # Volume adjustment

        # Convert text to speech and play immediately
        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print(f"Error occurred during playback: {e}")


# Example usage with increased speed (1.5x) and increased volume (0.8)
text_to_speech("gTTS response..increased speed", volume=0.8)

"""

"""

import ChatTTS
from IPython.display import Audio
import os
import time
import requests
import openai
import torchaudio
import torch
import numpy as np


chat = ChatTTS.Chat()
chat.load(compile=True)  # Set to True for better performance

say = "Just ask me a question ffs"

print(say)

texts = [
    say,
]

wavs = chat.infer(
    texts,
)

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)


def play_wav_file(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    # Wait until music has finished playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.quit()


# Example usage
play_wav_file("output1.wav")
"""

"""
import io
import asyncio
import edge_tts
from pydub import AudioSegment
from pydub.playback import play
import nest_asyncio

# Apply the nest_asyncio patch
nest_asyncio.apply()


async def text_to_speech_and_play(text, voice="en-GB-MiaNeural", speed=1):
    try:
        rate = "+" + str(int((speed - 1) * 100)) + "%"
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save("output.mp3")

        # Load the saved audio file
        sound = AudioSegment.from_file("output.mp3", format="mp3")

        # Play the audio
        play(sound)

    except Exception as e:
        print(f"Error occurred during playback: {e}")


# Example usage
text = "Hello, this is a test of the edge-tts library for text-to-speech conversion."
asyncio.run(text_to_speech_and_play(text, speed=1.2))  # Increase speed to 1.5x



"""
import torch

print(torch.__version__)  # Should print the installed PyTorch version
print(torch.cuda.is_available())  # Should return True if CUDA is available
