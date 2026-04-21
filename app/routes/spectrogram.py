import io
import matplotlib
matplotlib.use("Agg")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/spectrogram")
async def create_spectrogram(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)

    duration_sec = len(y) / sr

    D = librosa.stft(y)
    D = D[120:743, :]
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Match frontend exactly
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

    return StreamingResponse(buf, media_type="image/png")