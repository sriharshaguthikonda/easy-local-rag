import os
import asyncio
import numpy as np
import torch
from pydub import AudioSegment
from pydub.playback import play  # Importing play for audio playback
from io import BytesIO
import sys
from pathlib import Path

# Set up environment variables for eSpeak NG
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG"

# Add Kokoro directory to path
kokoro_path = Path(r"C:\Users\deletable\OneDrive\Kokoro-82M")
sys.path.insert(0, str(kokoro_path))

from models import build_model
from kokoro import generate

# Initialize model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.init()

MODEL = build_model(str(kokoro_path / "kokoro-v0_19.pth"), DEVICE)


async def generate_speech(text: str, voice_file: str, speed: float = 1.2):
    """Generate speech from text using a specific voice and play it."""
    try:
        # Load the specific voice pack
        VOICEPACK = torch.load(voice_file).to(DEVICE)

        # Generate audio tensor
        loop = asyncio.get_running_loop()
        audio, sr = await loop.run_in_executor(None, generate, MODEL, text, VOICEPACK)

        # Convert to PyDub segment
        audio_int16 = (audio * 32767).astype(np.int16)
        segment = AudioSegment(
            audio_int16.tobytes(), frame_rate=24000, sample_width=2, channels=1
        )

        # Adjust the playback speed
        segment = segment.speedup(playback_speed=speed)

        # Play the generated audio
        play(segment)

        # Save the audio file
        with open("output.wav", "wb") as f:
            segment.export(f, format="wav")

    finally:
        if "audio" in locals():
            del audio
        torch.cuda.empty_cache()


async def listen_to_all_voices(text: str):
    """Listen to all available voices."""
    voices_dir = kokoro_path / "voices"
    voice_files = voices_dir.glob(
        "*.pt"
    )  # Adjust if your voice files have different extensions

    for voice_file in voice_files:
        print(f"Listening to voice: {voice_file.name}")
        await generate_speech(text, str(voice_file), speed=1.4)  # Adjust speed here
        input(
            "Press Enter to continue to the next voice..."
        )  # Wait for user input before continuing


# Test generation of all voices
if __name__ == "__main__":
    text = "Hello! This is a test of Kokoro's text-to-speech capabilities."
    asyncio.run(listen_to_all_voices(text))
