import io
import numpy as np
import librosa
import tensorflow as tf
import uuid
import os
from fastapi import BackgroundTasks
from fastapi import APIRouter, UploadFile, File

jobs = {}
UPLOAD_DIR = "tmp"
os.makedirs(UPLOAD_DIR, exist_ok=True)



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


def process_file(job_id, file_path):
    jobs[job_id]["status"] = "processing"

    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()

        # your existing pipeline
        data = fourier_from_bytes(audio_bytes)
        times, scores = song_model_scores(data)
        detections = get_song_times(times, scores)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = detections

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)




@router.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}.wav")

    # save file to disk
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # initialize job
    jobs[job_id] = {"status": "queued"}

    # run in background
    background_tasks.add_task(process_file, job_id, file_path)

    return {"job_id": job_id}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    return jobs.get(job_id, {"status": "not_found"})

@router.get("/result/{job_id}")
def get_result(job_id: str):
    job = jobs.get(job_id)

    if not job:
        return {"error": "not_found"}

    if job["status"] != "done":
        return {"status": job["status"]}

    return {"detections": job["result"]}