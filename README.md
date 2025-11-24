# Voice-to-transcription-nemo
Transcription-mobile app
DMS NeMo Speech-to-Text API

A high-performance speech-to-text transcription service built using NVIDIA NeMo, featuring noise reduction, audio enhancement, chunk-based processing, and GPU-accelerated inference. This service exposes a simple FastAPI endpoint to convert WAV audio into text in real time.

📡 API Endpoint

Base URL:

http://192.168.1.47:8121/transcribe


Method: POST
Content-Type: multipart/form-data
Parameter:

file — WAV audio file

📌 Key Features
🎙️ Advanced Audio Processing

Noise reduction (Wiener filter + noisereduce)

Bandpass filtering (80 Hz – 8000 Hz)

Loudness normalization and gain boosting

Automatic conversion to mono

⚡ High-Accuracy Speech Recognition

NVIDIA NeMo stt_en_conformer_ctc_small model

GPU-accelerated inference (CUDA)

Handles noisy, low-quality, and long-duration audio

🔄 Scalable and Reliable Processing

Automatic 30-second chunking for long audio files

Memory-safe temp file handling

Rotating logs for long-term stability

🐳 Docker + GPU Ready

Runs inside the official NVIDIA NeMo Docker container with full GPU support.

📁 Project Directory Structure
.
├── voice2text-mobileapp.py       # Main FastAPI service
├── transcription.log             # Rotating log file
├── temp_chunks/                  # Auto-generated audio chunks
└── README.md                     # Documentation

🛠️ System Requirements
Hardware

NVIDIA GPU (recommended)

Minimum 6GB VRAM

Software

Ubuntu / Linux Server

Docker + NVIDIA Container Toolkit

NVIDIA Driver (CUDA-compatible)

🐳 Deployment Using NVIDIA NeMo Docker
1️⃣ Start NeMo Container
docker run -it --gpus all \
    -p 8121:8121 \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/nemo:25.02

2️⃣ Install Missing Dependencies

Inside the container:

pip install fastapi uvicorn pydub noisereduce soundfile scipy

▶️ Start the API Service

Inside the container:

python3 voice2text-mobileapp.py


You will see:

🌍 Starting FastAPI server on port 8121...

📤 Example API Request
Using cURL
curl -X POST "http://192.168.1.47:8121/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav"

Sample Response
{
  "transcript": "This is the transcribed text from your audio file."
}

🎧 Audio Processing Workflow

Each audio chunk goes through:

Wiener Noise Filtering

noisereduce Denoising

Bandpass Filtering

Amplitude Normalization

Gain Boost (+25%)

GPU-powered NeMo Transcription

This ensures clean and accurate transcripts even in noisy environments.

📝 Logging

Logs are stored in:

transcription.log


With automatic rotation:

Max size: 10 MB

5 backup log files

📦 Production Recommendations

Deploy using Docker Compose

Add Nginx reverse proxy

Use HTTPS (Certbot)

Monitor GPU usage (nvidia-smi)

Enable systemd auto-restart

❗ Notes

Only WAV audio format is supported (recommended: 16 kHz / mono).

Processing large audio files may take additional time due to chunking.

GPU is strongly recommended for real-time performance.
