import asyncio
import numpy as np
import torch
from pydub import AudioSegment
from io import BytesIO


# Add this at the very top of your script (groq_lama_chromadb_RAG_ETTS.py)
import sys
from pathlib import Path

# Point to your Kokoro-82M directory
kokoro_path = Path(r"C:\Users\deletable\OneDrive\Kokoro-82M")
sys.path.insert(0, str(kokoro_path))


from models import build_model  # From Kokoro repository
from kokoro import generate  # From Kokoro repository


# Pre-initialize model and voicepack (load once)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = build_model(r"C:\Users\deletable\OneDrive\Kokoro-82M\kokoro-v0_19.pth", DEVICE)
VOICEPACK = torch.load(
    r"C:\Users\deletable\OneDrive\Kokoro-82M/voices/af.pt", weights_only=True
).to(DEVICE)  # Default voice blend:cite[10]


async def text_to_speech_kokoro(
    text: str,
    speed: float = 1.2,
    volume: float = 1.0,
    lang: str = "en-us",
    voice: str = "af",
) -> BytesIO:
    """
    Convert text to speech using Kokoro TTS with real-time processing

    Args:
        text: Input text to synthesize
        speed: Playback speed multiplier (0.5-2.0)
        volume: Volume boost in dB (0.0-3.0)
        lang: Language code (en-us/en-gb)
        voice: Voice identifier or blend

    Returns:
        BytesIO buffer containing MP3 audio
    """
    try:
        if not text.strip():
            raise ValueError("Empty input text")

        # Generate raw audio using Kokoro's neural synthesis:cite[10]
        loop = asyncio.get_event_loop()
        audio, _ = await loop.run_in_executor(
            None, generate, MODEL, text, VOICEPACK, lang
        )

        # Convert to PyDub audio segment
        audio_int16 = (audio * 32767).astype(np.int16)
        segment = AudioSegment(
            audio_int16.tobytes(), frame_rate=24000, sample_width=2, channels=1
        )

        # Audio processing pipeline
        processed = segment.speedup(playback_speed=speed).apply_gain(volume * 10)

        # Export to MP3 buffer
        buffer = BytesIO()
        processed.export(buffer, format="mp3")
        buffer.seek(0)

        return buffer

    except Exception as e:
        print(f"TTS Error: {e}")
        raise
