#!/usr/bin/env python3
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import torch
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from nemo.collections.asr.models import ASRModel
import noisereduce as nr
from scipy.signal import butter, lfilter, wiener
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import tempfile

# ======================================
# CONFIGURATION
# ======================================
CURRENT_DIR = os.getcwd()  # current working directory
LOG_FILE_PATH = os.path.join(CURRENT_DIR, "transcription.log")
TEMP_DIR = os.path.join(CURRENT_DIR, "temp_chunks")
CHUNK_MS = 30000  # 30 seconds chunk
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(TEMP_DIR, exist_ok=True)  # ensure temp dir exists

# ======================================
# LOGGING CONFIGURATION
# ======================================
logger = logging.getLogger("transcription_api")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

if logger.hasHandlers():
    logger.handlers.clear()

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Rotating file handler
fh = RotatingFileHandler(LOG_FILE_PATH, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def log_flush():
    for handler in logger.handlers:
        handler.flush()

def log(msg, level="info"):
    if level == "debug":
        logger.debug(msg)
    elif level in ("warn", "warning"):
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.info(msg)
    log_flush()

# ======================================
# AUDIO PROCESSING UTILITIES
# ======================================
def bandpass_filter(data, sr, lowcut=80, highcut=8000, order=6):
    nyq = 0.5 * sr
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, data)

def reduce_noise_and_boost(wav_path):
    """
    Denoise, filter, normalize, and boost audio.
    This function reads audio, applies Wiener filter,
    noise reduction, bandpass filtering, normalization,
    and gain boosting.
    """
    audio, sr = sf.read(wav_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # convert to mono

    # Step 1: Wiener filter for noise smoothing
    audio_wiener = wiener(audio)

    # Step 2: Noise reduction using noisereduce
    audio_nr = nr.reduce_noise(y=audio_wiener, sr=sr, n_std_thresh_stationary=1.3)

    # Step 3: Bandpass filter
    audio_filtered = bandpass_filter(audio_nr, sr)

    # Step 4: Normalize
    max_val = np.max(np.abs(audio_filtered))
    if max_val > 0:
        audio_norm = audio_filtered / max_val * 0.98
    else:
        audio_norm = audio_filtered

    # Step 5: Boost gain (loudness)
    audio_boosted = np.clip(audio_norm * 1.25, -1.0, 1.0)

    # Overwrite wav file with processed audio
    sf.write(wav_path, audio_boosted, sr)
    return wav_path

# ======================================
# GPU INFO LOGGING
# ======================================
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    log(f"✅ GPU Detected: {gpu_name} ({gpu_mem:.2f} GB total)")
else:
    log("⚠️ No GPU detected — running on CPU", level="warning")

# ======================================
# LOAD MODEL ON STARTUP
# ======================================
log("🚀 Loading NeMo ASR model (stt_en_conformer_ctc_small)...")
try:
    model = ASRModel.from_pretrained("stt_en_conformer_ctc_small").to(DEVICE)
    log("✅ Model successfully loaded and ready for inference.")
except Exception as e:
    log(f"❌ Failed to load ASR model: {e}", level="error")
    sys.exit(1)

# ======================================
# FASTAPI APP
# ======================================
app = FastAPI(title="NeMo Audio Transcription API")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load audio and split into chunks
        audio = AudioSegment.from_wav(tmp_path)
        chunks = [audio[i:i + CHUNK_MS] for i in range(0, len(audio), CHUNK_MS)]
        log(f"🔹 Total chunks: {len(chunks)} for file '{file.filename}'")

        all_text = []
        for idx, chunk in enumerate(chunks):
            temp_path = os.path.join(TEMP_DIR, f"chunk_{idx}.wav")
            chunk.export(temp_path, format="wav")

            try:
                clean_path = reduce_noise_and_boost(temp_path)

                # Model inference - use GPU if available
                results = model.transcribe([clean_path])

                # Extract text safely
                if isinstance(results[0], str):
                    text = results[0]
                elif isinstance(results[0], dict) and "text" in results[0]:
                    text = results[0]["text"]
                elif hasattr(results[0], "text"):
                    text = results[0].text
                else:
                    text = str(results[0])

                log(f"✅ Chunk {idx + 1}/{len(chunks)} transcribed successfully.")
                all_text.append(text.strip())

            except Exception as e:
                log(f"⚠️ Error in chunk {idx + 1}: {e}", level="error")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        final_text = "\n".join(all_text)
        os.remove(tmp_path)

        log(f"📝 Transcription completed for file: {file.filename}")
        return JSONResponse({"transcript": final_text})

    except Exception as e:
        log(f"❌ Transcription error: {e}", level="error")
        return JSONResponse({"error": str(e)}, status_code=500)

# ======================================
# START SERVER
# ======================================
if __name__ == "__main__":
    log("🌍 Starting FastAPI server on port 8121...")
    uvicorn.run(app, host="0.0.0.0", port=8121)
