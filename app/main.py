from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

app = FastAPI()

# ---- CORS setup ----
origins = [
    "http://localhost:5173",  # Svelte dev server
    "http://127.0.0.1:5173",  # just in case
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # allow your frontend
    allow_credentials=True,
    allow_methods=["*"],     # allow all HTTP methods
    allow_headers=["*"],     # allow all headers
)

@app.post("/spectrogram")
async def create_spectrogram(file: UploadFile = File(...)):
    # 1️⃣ Read the uploaded audio
    audio_bytes = await file.read()
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # 2️⃣ Generate STFT and convert to dB
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # 3️⃣ Plot spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Spectrogram")

    # 4️⃣ Save figure to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    # 5️⃣ Return as PNG response
    return StreamingResponse(buf, media_type="image/png")