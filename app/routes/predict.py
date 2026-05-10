import io
import numpy as np
import librosa
import tensorflow as tf
import uuid
import os
from fastapi import BackgroundTasks
from fastapi import APIRouter, UploadFile, File
import time
import json
from pathlib import Path
import shutil
from datetime import datetime, timedelta

jobs = {}

DATA_DIR = Path(os.getenv("DATA_DIR", "tmp"))
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

router = APIRouter()

# ---- load model ONCE (important) ----
# model = tf.keras.models.load_model("app/ml/ml_songs.model")
MODEL_PATH = "app/ml/ml_songs_v1.model"

model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

time_converter = 0.023219814

def cleanup_old_jobs(max_age_hours=48):
    cutoff = datetime.now() - timedelta(hours=max_age_hours)

    for job_path in JOBS_DIR.iterdir():

        if not job_path.is_dir():
            continue

        try:
            modified_time = datetime.fromtimestamp(
                job_path.stat().st_mtime
            )

            if modified_time < cutoff:
                print(f"Deleting old job: {job_path}")
                shutil.rmtree(job_path)

        except Exception as e:
            print(f"Cleanup failed for {job_path}: {e}")

def has_active_job():
    for job in jobs.values():
        if job.get("status") in ["queued", "processing"]:
            return True
    return False

def job_dir(job_id):
    path = JOBS_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_status_path(job_id):
    return job_dir(job_id) / "status.json"


def write_job(job_id, data):
    jobs[job_id] = data
    with open(job_status_path(job_id), "w") as f:
        json.dump(data, f)


def read_job(job_id):
    if job_id in jobs:
        return jobs[job_id]

    path = job_status_path(job_id)
    if not path.exists():
        return None

    with open(path, "r") as f:
        data = json.load(f)

    jobs[job_id] = data
    return data

def compute_intro_threshold(array):
    array_median = np.median(array)
    return np.median(array) + np.std(array) * (0.75 + (abs(-45 - array_median) / 20)) + 5


def fourier_from_bytes(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    D = librosa.stft(y)
    D = D[120:743, :]
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db



def timed_step(name, fn):
    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    print(f"[TIMING] {name}: {end - start:.3f} sec")
    return result


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



def get_song_times_from_all(all_times, all_scores, score_threshold=0.9, min_plateau_len=3):
    strength_list = []
    song_time_list = []

    plateau_length = 0
    plateau_start = 0

    for i in range(len(all_times) - 3):
        if all_scores[i] > score_threshold:
            if plateau_length == 0:
                plateau_start = i
            plateau_length += 1
        else:
            if plateau_length >= min_plateau_len:
                song_time = ((i + plateau_start) / 2) * (time_converter * 2)
                song_time_list.append(song_time)
                strength_list.append(plateau_length)
            plateau_length = 0

    return np.array(song_time_list), np.array(strength_list)


def remove_bad_timing_songs(song_time_list, strength_list):
    if len(song_time_list) == 0:
        return np.array([])

    if len(song_time_list) == 1:
        return np.array(song_time_list)

    new_list = []
    diffs = []

    for i in range(len(song_time_list)):
        if i > 0:
            diffs.append(song_time_list[i] - song_time_list[i - 1])
        else:
            diffs.append(0)

    median_rate = np.median(diffs[1:]) if len(diffs) > 1 else 0
    timing_cutoff = median_rate * 0.6
    sandwich_cutoff = median_rate * 0.6

    for i in range(len(song_time_list)):
        # remove sandwiched close songs
        if i > 0 and i < len(song_time_list) - 1:
            if diffs[i] < sandwich_cutoff and diffs[i + 1] < sandwich_cutoff:
                continue

        # remove preceding close weak songs
        if i < len(song_time_list) - 1:
            if diffs[i + 1] < timing_cutoff and strength_list[i] <= 5 and strength_list[i + 1] >= 6:
                continue

        # remove following close weak songs
        if i > 0:
            if diffs[i] < timing_cutoff and strength_list[i] <= 5 and strength_list[i - 1] >= 6:
                continue

        new_list.append(song_time_list[i])

    return np.array(new_list)


def reinsert_weak_songs(all_times, all_scores, song_times):
    song_times = list(song_times)

    if len(song_times) < 2:
        return np.array(sorted(song_times))

    diffs = []
    for i in range(len(song_times)):
        if i > 0:
            diffs.append(song_times[i] - song_times[i - 1])
        else:
            diffs.append(0)

    median_rate = np.median(diffs[1:]) if len(diffs) > 1 else 0
    jump_to_i = 0

    for i in range(len(all_times) - 20):
        if i <= jump_to_i:
            continue

        too_close = False
        for verified_time in song_times:
            if abs(all_times[i] - verified_time) < max(median_rate - 1.5, 0):
                too_close = True
                break

        if too_close:
            continue

        for verified_time in song_times:
            if i > 20 and abs(all_times[i] - verified_time) < median_rate + 1:
                local_max = np.max(all_scores[i - 20:i + 20])
                if all_scores[i] > 0.2 and all_scores[i] >= local_max:
                    song_times.append(all_times[i])
                    jump_to_i = i + 20
                    break

    return np.array(sorted(song_times))


def refine_song_events(times, scores):
    raw_song_times, strengths = get_song_times_from_all(times, scores)

    if len(raw_song_times) == 0:
        return {
            "raw_candidate_times": [],
            "cleaned_song_times": [],
            "strengths": []
        }

    trimmed_song_times = remove_bad_timing_songs(raw_song_times, strengths)
    reinserted_song_times = reinsert_weak_songs(times, scores, trimmed_song_times)

    return {
        "raw_candidate_times": raw_song_times.tolist(),
        "cleaned_song_times": reinserted_song_times.tolist(),
        "strengths": strengths.tolist()
    }



#creates a post introductory pooled spectrogram
def post_pool(array, column, end_column, rows, columns, cell_height, cell_length):
    row_index = 0
    post_note_array = []
    if column + 70 > columns:
        end_column = columns
    while row_index < rows - cell_height:
        col_index = 0
        row_values = []
        while column + col_index <= end_column - cell_length:
            row_values.append(np.mean(array[row_index:row_index + cell_height,column + col_index:column + col_index + cell_length]))
            col_index += cell_length
        post_note_array.append(row_values)
        row_index += cell_height
    return np.array(post_note_array)

def filter_one_pool_new(pool, range_threshold, range_range, horizontals = False):
    thisPool = pool.copy()

    #look for stacks of 3 cells that are similar to each other and remove them
    indexes = []
    for i in range(len(thisPool)):
        for j in range(len(thisPool[0])):
            if i < range_range:
                bot = 0
                top = range_range * 2
            elif i > len(thisPool) - range_range:
                bot = len(thisPool) - 2 * range_range
                top = len(thisPool)
            else:
                top = i + range_range
                bot = i - range_range
            #vol_range = np.max(thisPool[bot:top,j]) - np.min(thisPool[bot:top,j])
            
            
            if i <= len(thisPool) - range_range and (abs(thisPool[i, j] - np.min(thisPool[i:top+1, j])) < range_threshold):
                indexes.append([i, j])
            if i >= range_range and (abs(thisPool[i, j] - np.min(thisPool[bot:i, j])) < range_threshold):
                indexes.append([i, j])
            
            # if vol_range < range_threshold:
            #     indexes.append([i, j])
    for index in indexes:
        thisPool[index[0], index[1]] = -80

    #remove anything quieter than a given amount
    for i in range(len(thisPool)):
        for j in range(len(thisPool[0])):
            if thisPool[i,j] < -65:
                thisPool[i,j] = -80
    
    #remove any full horizontal lines                
    if horizontals:
        for i in range(len(thisPool)):
            if np.sort(thisPool[i,:])[2] > -80:
                thisPool[i, :] = -80
    return thisPool

def find_loudest_box(array, width, height):
    norm_array = array + 80
    rows = len(array)
    cols = len(array[0])
    best_sum = 0
    best_i = 0
    best_j = 0
    for i in range(rows):
        for j in range(cols):
            if i < rows - height and j < cols - width:
                box_sum = np.sum(norm_array[i:i+height,j:j+width])
                # print(box_sum, i, j)
                if box_sum >= best_sum:
                    best_sum = box_sum
                    best_i = i
                    best_j = j
    best_i_perc = best_i/rows
    best_j_perc = best_j/cols
    return best_i, best_j, best_i_perc, best_j_perc

def filter_non_box(array, best_i, best_j, width, height):
    new_array = array.copy()
    rows = len(array)
    cols = len(array[0])
    for i in range(rows):
        for j in range(cols):
            if i < best_i or i >= best_i + height or j < best_j or j >= best_j + width:
                new_array[i, j] = -80
            # if i == best_i or i==best_i + height or j == best_j or j == best_j + width:
            #     new_array[i, j] = 0
    return new_array

#this intro note finder uses a window to move up and down the region searching for the brightest, leftmost, line.
def intro_note_finder(array, box_i_perc, box_j_perc, intro_threshold):
    rows = len(array)
    cols = len(array[0])
    box_length = 10
    box_height = 15
    box_i = int(box_i_perc * rows)
    box_j = int(box_j_perc * cols)
    search_zone_bot = max(0, box_i - 30)
    search_zone_top = min(rows, box_i + 80)
    search_zone_left = max(0, box_j - 20)
    search_zone_right = box_j + 2 * box_length

    best_window_score = -100000000
    best_i = 0
    best_j = 0
    mean_diffs = []
    line_vols = []
    scores = []
    js = []
    for i in range(search_zone_bot, search_zone_top - box_height):
        for j in range(search_zone_left, search_zone_right + 1 - box_length):
            window_array = array[i:i+box_height,j:j+box_length]
            if np.max(window_array[5:10,0]) > intro_threshold:
                mean_diffs.append(0)
                line_vols.append(0)
                scores.append(0)
                js.append(j)
                continue
            maxes = []
            col_diffs = []
            for col in range(box_length):
                maxes.append(np.max(window_array[5:10, col]) + 80)
            line_vols.append(np.mean(maxes))
            if np.median(maxes) < intro_threshold:
                mean_diffs.append(0)
                scores.append(0)
                js.append(j)
                continue
            for col in range(box_length):
                col_diffs.append( (np.max(window_array[5:10, col]) - np.median(window_array[11:15])) + (np.max(window_array[5:10, col]) - np.median(window_array[0:4])) )
            mean_diff = np.mean(col_diffs)
            mean_diffs.append(mean_diff/2)
            score = mean_diff - 0.7 * abs(j - (box_j - 15))
            scores.append(score)
            js.append(j)
            if score > best_window_score:
                best_window_score = score
                best_i = i
                best_j = j

    final_j = best_j
    max_i = np.argmax(array[best_i + 5: best_i + 10, final_j])
    final_i = best_i + 5 + max_i
    final_j_perc = final_j * cols
    final_i_perc = final_i * rows
    return final_i, final_j, final_i_perc, final_j_perc

def highlight_intro_note(array, intro_i, intro_j):
    new_array = array.copy()
    rows = len(array)
    cols = len(array[0])
    for i in range(rows):
        for j in range(cols):
            if (i == intro_i and j == intro_j):
                new_array[i, j] = 100
            else:
                new_array[i, j] = new_array[i, j] + 80
    return new_array

def convert_start_i_and_j(time_og, start_i, start_j):
    start_time_of_array = time_og - 20* time_converter
    row_ratio = 623/124
    new_row = int(start_i * row_ratio)
    time_of_note = start_time_of_array + start_j * time_converter
    start_freq = (new_row+120)*10.7666
    return new_row, start_j, start_freq, time_of_note

def get_start_times_and_freqs_from_pools_with_displays(data, song_time_list, time_value_list, intro_threshold):
    shaved_list = []
    if len(time_value_list) > 0:
        for song_time in song_time_list:
            for compare_time in time_value_list:
                if abs(song_time - compare_time) < 1.5:
                    shaved_list.append(song_time)
    else:
        shaved_list = song_time_list
        
    start_times = []
    start_freqs = []
    for time in shaved_list:
        test_pool = post_pool(data, int(time/time_converter)-20, int(time/time_converter) + 70, len(data), len(data[0]), 5, 7)
        time_pool = post_pool(data, int(time/time_converter)-20, int(time/time_converter) + 70, len(data), len(data[0]), 5, 1)


        filtered_pool = filter_one_pool_new(test_pool, 6, 4, True)    
        # lightly_filtered_pool = filter_one_pool_new(time_pool, 5, 4, False)
        box_i, box_j, box_i_perc, box_j_perc = find_loudest_box(filtered_pool, 6, 60)
        # only_box_pool = filter_non_box(filtered_pool, box_i, box_j, 6, 60)
        intro_i, intro_j, intro_i_perc, intro_j_perc = intro_note_finder(time_pool, box_i_perc, box_j_perc, intro_threshold)            
        # just_start_note_array = highlight_intro_note(time_pool, intro_i, intro_j)

        big_i, big_j, song_freq, time_of_note = convert_start_i_and_j(time, intro_i, intro_j)

        start_times.append(time_of_note)
        start_freqs.append(song_freq)
    return np.array(start_times), np.array(start_freqs)

def build_detection_boxes(data, cleaned_song_times, intro_threshold):
    start_times, start_freqs = get_start_times_and_freqs_from_pools_with_displays(
        data,
        cleaned_song_times,
        [],
        intro_threshold
    )

    detections = []
    for t, f in zip(start_times, start_freqs):
        detections.append({
            "time": float(t),
            "low_freq_hz": float(f - 200),
            "high_freq_hz": float(f + 200),
        })

    return detections


# def process_file(job_id, file_path):
#     jobs[job_id]["status"] = "processing"

#     try:
#         with open(file_path, "rb") as f:
#             audio_bytes = f.read()

#         # your existing pipeline
#         data = fourier_from_bytes(audio_bytes)
#         intro_threshold = compute_intro_threshold(data)
#         times, scores = song_model_scores(data)

#         refined = refine_song_events(times, scores)
#         detections = build_detection_boxes(data, refined["cleaned_song_times"], intro_threshold)

#         jobs[job_id]["status"] = "done"
#         jobs[job_id]["result"] = {
#             "raw_scores_count": len(scores),
#             "raw_candidate_times": refined["raw_candidate_times"],
#             "cleaned_song_times": refined["cleaned_song_times"],
#             "detections": detections
#         }

#     except Exception as e:
#         jobs[job_id]["status"] = "error"
#         jobs[job_id]["error"] = str(e)


def process_file(job_id, file_path):
    import time
    total_start = time.perf_counter()

    job = read_job(job_id) or {}
    job["status"] = "processing"
    write_job(job_id, job)

    try:
        # ---- Read file ----
        audio_bytes = timed_step("read_file", lambda: open(file_path, "rb").read())

        # ---- Pipeline ----
        data = timed_step("fourier_from_bytes", lambda: fourier_from_bytes(audio_bytes))

        intro_threshold = timed_step(
            "compute_intro_threshold",
            lambda: compute_intro_threshold(data)
        )

        times, scores = timed_step(
            "song_model_scores",
            lambda: song_model_scores(data)
        )

        refined = timed_step(
            "refine_song_events",
            lambda: refine_song_events(times, scores)
        )

        detections = timed_step(
            "build_detection_boxes",
            lambda: build_detection_boxes(
                data,
                refined["cleaned_song_times"],
                intro_threshold
            )
        )

        # ---- Total time ----
        total_end = time.perf_counter()
        print(f"[TIMING] TOTAL: {total_end - total_start:.3f} sec")

        job = read_job(job_id) or {}
        job["status"] = "done"
        job["result"] = {
            "raw_scores_count": len(scores),
            "raw_candidate_times": refined["raw_candidate_times"],
            "cleaned_song_times": refined["cleaned_song_times"],
            "detections": detections
        }
        write_job(job_id, job)

    except Exception as e:
        job = read_job(job_id) or {}
        job["status"] = "error"
        job["error"] = str(e)
        write_job(job_id, job)


@router.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    cleanup_old_jobs()
    if has_active_job():
        return {
            "error": "server_busy",
            "message": "Another audio file is currently being processed. Please try again later."
        }

    job_id = str(uuid.uuid4())
    file_path = job_dir(job_id) / "input.wav"

    # save file to disk
    audio_bytes = await file.read()

    if len(audio_bytes) > MAX_UPLOAD_SIZE_BYTES:
        return {
            "error": "file_too_large",
            "message": f"File is too large. Maximum size is {MAX_UPLOAD_SIZE_MB} MB."
        }

    # save file to disk
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    # initialize job
    write_job(job_id, {
        "status": "queued",
        "file_path": str(file_path)
    })

    # run in background
    background_tasks.add_task(process_file, job_id, file_path)

    return {"job_id": job_id}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    job = read_job(job_id)
    if not job:
        return {"status": "not_found"}

    return {"status": job.get("status", "unknown")}


@router.get("/result/{job_id}")
def get_result(job_id: str):
    job = read_job(job_id)

    if not job:
        return {"error": "not_found"}

    if job.get("status") != "done":
        return {"status": job.get("status")}

    return job.get("result", {})