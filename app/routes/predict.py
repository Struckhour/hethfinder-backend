import io
import numpy as np
import librosa
import tensorflow as tf

from fastapi import APIRouter, UploadFile, File

router = APIRouter()

# ---- load model ONCE (important) ----
# model = tf.keras.models.load_model("app/ml/ml_songs.model")
MODEL_PATH = "app/ml/ml_songs_v1.model"

model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

time_converter = 0.023219814


def fourier_from_bytes(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    D = librosa.stft(y)
    D = D[120:743, :]
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db


def song_predict(array, column, columns):
    if column > len(array[0]) - 69:
        new_array = array[:, column:columns]
        shorter_cols = np.shape(new_array)[1]

        zero_array = np.full((623, 69), -80)
        zero_array[:, :shorter_cols] = new_array
        song_array = zero_array
    else:
        song_array = array[:, column:column+69]

    song_array = (song_array + 80) * (255 / 80)
    song_array = song_array.reshape(-1, 623, 69, 1)

    x_min = song_array.min(axis=(1, 2), keepdims=True)
    x_max = song_array.max(axis=(1, 2), keepdims=True)
    song_array = (song_array - x_min) / (x_max - x_min + 1e-8)

    # prediction = model.predict(song_array, verbose=0)
    prediction = infer(conv2d_input=song_array)
    prediction = prediction["activation_5"].numpy()
    return prediction[0][0]



def song_model_scores(data):
    all_times = []
    all_scores = []

    i = 0
    columns = len(data[0])

    while i < columns:
        score = song_predict(data, i, columns)
        all_scores.append(score)
        all_times.append(i * time_converter)
        i += 2

    return np.array(all_times), np.array(all_scores)


def get_song_times(all_times, all_scores):
    song_times = []

    for i in range(len(all_scores) - 3):
        if all_scores[i] > 0.9:
            song_times.append(all_times[i])

    return song_times


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # 1. spectrogram
    data = fourier_from_bytes(audio_bytes)

    # 2. model inference
    times, scores = song_model_scores(data)

    # 3. extract detections
    detections = get_song_times(times, scores)

    return {
        "detections": detections[:50]  # limit for sanity
    }