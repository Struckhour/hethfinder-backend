import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/spectrogram")
async def create_spectrogram(file: UploadFile = File(...)):
    # 1️⃣ Read uploaded audio
    audio_bytes = await file.read()
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # 2️⃣ Generate STFT and convert to dB
    D = librosa.stft(y)
    D = D[120:743, :]
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # 3️⃣ Plot spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, ax=ax, cmap="Greys")
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 4️⃣ Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # 5️⃣ Return as PNG response
    return StreamingResponse(buf, media_type="image/png")