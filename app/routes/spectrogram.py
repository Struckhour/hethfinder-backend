import io
import matplotlib
matplotlib.use("Agg")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

def make_spectrogram_buffer(audio_bytes: bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)

    duration_sec = len(y) / sr

    D = librosa.stft(y)
    D = D[120:743, :]
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    pixels_per_second = 40
    min_width_px = 800
    height_px = 450
    dpi = 100

    width_px = max(int(duration_sec * pixels_per_second), min_width_px)

    fig_width_in = width_px / dpi
    fig_height_in = height_px / dpi

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis=None,
        y_axis=None,
        ax=ax,
        cmap="Greys"
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    return buf

router = APIRouter()

@router.post("/spectrogram")
async def create_spectrogram(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    if len(audio_bytes) > MAX_UPLOAD_SIZE_BYTES:
        return {
            "error": "file_too_large",
            "message": f"File is too large. Maximum size is {MAX_UPLOAD_SIZE_MB} MB."
        }

    buf = make_spectrogram_buffer(audio_bytes)
    return StreamingResponse(buf, media_type="image/png")


@router.get("/spectrogram/{job_id}")
def get_spectrogram_for_job(job_id: str):
    file_path = Path("tmp") / "jobs" / job_id / "input.wav"

    if not file_path.exists():
        return {"error": "file_not_found"}

    audio_bytes = file_path.read_bytes()
    buf = make_spectrogram_buffer(audio_bytes)

    return StreamingResponse(buf, media_type="image/png")