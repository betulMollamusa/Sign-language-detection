import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

# ---------------------
# Sabitler ve Yardımcılar
# ---------------------

FONT_PATH = "arial.ttf" 
FONT_SIZE = 32
TEXT_BG_COLOR = (30, 30, 30, 180)
TEXT_COLOR = (0, 255, 0)
TEXT_SHADOW_COLOR = (0, 0, 0)

def draw_text(frame, text, pos=(10, 40), font_size=FONT_SIZE, color=TEXT_COLOR, bg=True):
    font = ImageFont.truetype(FONT_PATH, font_size)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil, 'RGBA')

    x, y = pos
    # Arka plan kutusu çiz
    if bg:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        padding = 5
        rect_bg = (x - padding, y - padding, x + w + padding, y + h + padding)
        draw.rectangle(rect_bg, fill=TEXT_BG_COLOR)

    # Gölge (shadow) için aynı metni biraz offset ile siyah yaz
    shadow_pos = (x + 2, y + 2)
    draw.text(shadow_pos, text, font=font, fill=TEXT_SHADOW_COLOR)

    # Ana metni yaz
    draw.text(pos, text, font=font, fill=color)

    return np.array(img_pil)

# ---------------------
# Etiketleri ve Modeli Yükle
# ---------------------

with open("label.csv", 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

data = [line.strip().replace('"', '').split(';') for line in lines]
label_df = pd.DataFrame(data, columns=['class_id', 'class_name'])
label_df['class_id'] = label_df['class_id'].astype(int)
label_map = dict(zip(label_df['class_id'], label_df['class_name']))

MODEL_PATH = "newest_model_fp16.tflite"
assert Path(MODEL_PATH).exists(), f"Model dosyası bulunamadı: {MODEL_PATH}"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("TFLite FP16 model yüklendi →", MODEL_PATH)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SHAPE = tuple(input_details[0]['shape'])
SEQ_LEN, FEAT_DIM = INPUT_SHAPE[1], INPUT_SHAPE[2]
assert (SEQ_LEN, FEAT_DIM) == (30, 225), f"Model input {INPUT_SHAPE[1:]} 30×225 değil!"

INPUT_DTYPE = input_details[0]['dtype']
OUTPUT_DTYPE = output_details[0]['dtype']

# ---------------------
# MediaPipe Ayarları
# ---------------------

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# ---------------------
# Tahmin Fonksiyonu
# ---------------------

def tflite_predict(seq: np.ndarray) -> np.ndarray:
    seq = seq.astype(INPUT_DTYPE, copy=False)
    interpreter.set_tensor(input_details[0]['index'], seq)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index']).astype(np.float32)
    return probs

# ---------------------
# Ana Döngü
# ---------------------

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı!")

frame_buffer = []
action_history = []
last_prediction_time = 0.0
PREDICTION_INTERVAL = 3.0
is_predicting = False

fps_calc_time = time.time()
frame_count = 0
fps = 0

print("\n▶ Kameradan görüntü alınıyor... (B: Başlat, E: Bitir, Q: Çıkış)\n")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Kamera akışı kayboldu.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hands_res = hands.process(frame_rgb)
    pose_res = pose.process(frame_rgb)
    face_res = face.process(frame_rgb)

    feat = []

    # Pose landmark
    if pose_res.pose_landmarks:
        for lm in pose_res.pose_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
        mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
    else:
        feat.extend([0.0] * (33 * 3))

    # Hands
    if hands_res.multi_hand_landmarks:
        for h_lm in hands_res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, h_lm, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 215, 0), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(255, 165, 0), thickness=2))
            for lm in h_lm.landmark:
                feat.extend([lm.x, lm.y, lm.z])
        if len(hands_res.multi_hand_landmarks) == 1:
            feat.extend([0.0] * (21 * 3))
    else:
        feat.extend([0.0] * (21 * 3 * 2))

    # Face mesh - ilk 10 landmark
    if face_res.multi_face_landmarks:
        for lm in face_res.multi_face_landmarks[0].landmark[:10]:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * (10 * 3))

    if len(feat) < FEAT_DIM:
        feat.extend([0.0] * (FEAT_DIM - len(feat)))
    elif len(feat) > FEAT_DIM:
        feat = feat[:FEAT_DIM]

    feat = np.array(feat, dtype=np.float32)
    max_val = np.max(np.abs(feat))
    if max_val != 0:
        feat = feat / max_val

    if is_predicting:
        frame_buffer.append(feat.tolist())
        if len(frame_buffer) > SEQ_LEN:
            frame_buffer.pop(0)

        if len(frame_buffer) >= 10 and (time.time() - last_prediction_time) > PREDICTION_INTERVAL:
            buf = frame_buffer.copy()
            while len(buf) < SEQ_LEN:
                buf.append(buf[-1])
            inp = np.array(buf[-SEQ_LEN:]).reshape(1, SEQ_LEN, FEAT_DIM)
            probs = tflite_predict(inp)
            cls = int(np.argmax(probs))
            action_history.append(cls)
            last_prediction_time = time.time()

        if action_history:
            most_common_cls = Counter(action_history[-10:]).most_common(1)[0][0]
            label = label_map.get(most_common_cls, "Bilinmeyen")
            pred_text = f"Tahmin: {label} ({most_common_cls})"
        else:
            pred_text = "Veri toplanıyor..."
    else:
        pred_text = "Tahmin beklemede (B: Başlat, E: Bitir)"

    # FPS hesaplama
    frame_count += 1
    if (time.time() - fps_calc_time) > 1.0:
        fps = frame_count / (time.time() - fps_calc_time)
        fps_calc_time = time.time()
        frame_count = 0

    # Üst panel çizimi (yarı şeffaf kutu)
    overlay = frame.copy()
    panel_height = 90
    cv2.rectangle(overlay, (0,0), (frame.shape[1], panel_height), (20,20,20), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Tahmin metni
    frame = draw_text(frame, pred_text, pos=(10, 10), font_size=28, color=(0, 255, 0), bg=False)

    # FPS göstergesi
    fps_text = f"FPS: {fps:.1f}"
    frame = draw_text(frame, fps_text, pos=(frame.shape[1] - 140, 10), font_size=24, color=(255, 255, 0), bg=False)

    # Kullanıcı kısayolları (sağ alt köşede)
    keys_text = "B: Başlat  E: Bitir  Q: Çıkış"
    frame = draw_text(frame, keys_text, pos=(10, frame.shape[0] - 40), font_size=24, color=(180, 180, 180), bg=True)

    # Son tahminlerden kısaca mini gösterim (sol alt köşe)
    hist_len = 5
    recent = action_history[-hist_len:] if len(action_history) >= hist_len else action_history
    for i, cls_id in enumerate(recent):
        txt = label_map.get(cls_id, "?")
        pos_x = 10 + i * 120
        pos_y = frame.shape[0] - 90
        txt_disp = f"{txt}"
        frame = draw_text(frame, txt_disp, pos=(pos_x, pos_y), font_size=22, color=(0, 200, 255), bg=True)

    # Görüntüyü göster
    scale_percent = 150  # %150 büyüt (1.5 kat)
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("İşaret Dili Tanıma", frame_resized)

    #cv2.imshow("İşaret Dili Tanıma", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        is_predicting = True
        frame_buffer.clear()
        action_history.clear()
    elif key == ord('e'):
        is_predicting = False
        frame_buffer.clear()

cap.release()
cv2.destroyAllWindows()
print("Çıkıldı.")
