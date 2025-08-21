import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_holistic = mp.solutions.holistic

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks = []

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=0) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(frame_rgb)
            frame_landmark = []

            if result.pose_landmarks:
                frame_landmark.extend([coord for lm in result.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z)])
            else:
                frame_landmark.extend([0] * 33 * 3)

            if result.right_hand_landmarks:
                frame_landmark.extend([coord for lm in result.right_hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)])
            else:
                frame_landmark.extend([0] * 21 * 3)

            if result.left_hand_landmarks:
                frame_landmark.extend([coord for lm in result.left_hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)])
            else:
                frame_landmark.extend([0] * 21 * 3)

            landmarks.append(frame_landmark)

    cap.release()
    return np.array(landmarks)


def process_video(args):
    video_path, output_path = args
    try:
        landmarks = extract_landmarks(video_path)
        np.save(output_path, landmarks)
        return None  # Başarılı
    except Exception as e:
        return f"{os.path.basename(video_path)} → {str(e)}"


def load_labels_and_process(split):
    base_dir = "first_100/"
    labels_path = os.path.join(base_dir, f"{split}_labels_first_100.csv")
    video_dir = os.path.join(base_dir, f"{split}_first_100_classes")
    output_dir = os.path.join(base_dir, "processed_data_100", split)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(labels_path, header=None, names=["video_id", "label"])
    tasks = []

    for _, row in df.iterrows():
        video_file = f"{row['video_id']}_color.mp4"
        video_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, f"{row['video_id']}_{row['label']}.npy")

        if not os.path.exists(output_path):
            tasks.append((video_path, output_path))

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_video, tasks), total=len(tasks), desc=f"{split.upper()} işlemi"))

    errors = list(filter(None, results))
    if errors:
        print(f"\n{split.upper()} → Hatalı dosyalar:")
        for err in errors:
            print(err)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        load_labels_and_process(split)
    print("Tüm işlemler tamamlandı.")
