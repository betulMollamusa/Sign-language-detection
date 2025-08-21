"""
İşaret Dili Tanıma — Görsel Arayüzlü Sürüm
-------------------------------------------------
• cv2 + MediaPipe ile landmark çıkarır.
• 30×225'lik kare tamponu hazırlar ve eğitilmiş Keras modeliyle tahmin yapar.
• FPS, kısayollar ve son tahmin geçmişi dâhil gelişmiş OSD (On‑Screen Display) gösterir.
• Kontroller: B → Başlat, E → Bitir, Q → Çıkış
* Eğitim sırasında her kare kendi maksimumuna göre normalize edildiği için
  tahmin sırasında da aynı mantık uygulanır.
"""

import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

# ------------------------------------------------------------
# 1. Sabitler ve Yardımcı Fonksiyonlar
# ------------------------------------------------------------

FONT_PATH = "arial.ttf"  # Türkçe karakter destekli font dosyası
FONT_SIZE = 32
TEXT_BG_COLOR = (30, 30, 30, 180)   # (R,G,B,A)
TEXT_COLOR = (0, 255, 0)             # BGR!
TEXT_SHADOW_COLOR = (0, 0, 0)        # Gölge rengi (BGR)


def draw_text(frame: np.ndarray, text: str, *, pos: tuple[int, int] = (10, 40),
              font_size: int = FONT_SIZE, color: tuple[int, int, int] = TEXT_COLOR,
              bg: bool = True) -> np.ndarray:
    """PIL ile BGR frame üzerine (Türkçe karakter destekli) metin çizer.

    Args:
        frame: cv2 (BGR) görüntü.
        text: Yazılacak metin.
        pos: Sol‑üst köşe koordinatı (x, y).
        font_size: Font boyutu.
        color: Metin rengi (BGR).
        bg: Arka plan yarı saydam kutu çizilsin mi?
    Returns:
        Metin çizilmiş yeni frame (np.ndarray).
    """

    # Fontu yükle (bulunamazsa varsayılan fonta düşer)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        font = ImageFont.load_default()

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil, "RGBA")

    x, y = pos

    # Arka plan kutusu
    if bg:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        padding = 5
        rect = (x - padding, y - padding, x + w + padding, y + h + padding)
        draw.rectangle(rect, fill=TEXT_BG_COLOR)

    # Gölge
    shadow_offset = (2, 2)
    draw.text((x + shadow_offset[0], y + shadow_offset[1]), text, font=font,
              fill=TEXT_SHADOW_COLOR)

    # Ana metin
    draw.text(pos, text, font=font, fill=color[::-1])  # PIL RGB, OpenCV BGR — ters çeviriyoruz

    return np.array(img_pil)

# ------------------------------------------------------------
# 2. Etiket Haritası
# ------------------------------------------------------------

with open("label.csv", "r", encoding="utf-8-sig") as f:
    lines = f.readlines()

data = [line.strip().replace('"', "").split(';') for line in lines]
label_df = pd.DataFrame(data, columns=["class_id", "class_name"])
label_df["class_id"] = label_df["class_id"].astype(int)
label_map = dict(zip(label_df["class_id"], label_df["class_name"]))

# ------------------------------------------------------------
# 3. Model Yükle
# ------------------------------------------------------------

MODEL_PATH = "newest_model.keras"
assert Path(MODEL_PATH).exists(), f"Model dosyası bulunamadı: {MODEL_PATH}"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Keras model yüklendi →", MODEL_PATH)

INPUT_SHAPE = model.input_shape  # (None, 30, 225)
assert INPUT_SHAPE[1:] == (30, 225), \
    f"Model input shape {INPUT_SHAPE[1:]} — beklenen (30, 225)"

SEQ_LEN, FEAT_DIM = 30, 225

# ------------------------------------------------------------
# 4. MediaPipe Ayarları
# ------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# ------------------------------------------------------------
# 5. Döngü Değişkenleri
# ------------------------------------------------------------

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı!")

frame_buffer: list[list[float]] = []
pred_history: list[int] = []
last_pred_time = 0.0
PRED_INTERVAL = 2.5  # sn
is_predicting = False

# FPS ölçümü
fps_start = time.time()
frame_counter = 0
fps = 0.0

print("\n▶ Kameradan görüntü alınıyor… (B: Başlat, E: Bitir, Q: Çıkış)\n")

# ------------------------------------------------------------
# 6. Ana Döngü
# ------------------------------------------------------------

while True:
    ok, frame = cap.read()
    if not ok:
        print("Kamera akışı kayboldu.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe inference
    hands_res = hands.process(frame_rgb)
    pose_res = pose.process(frame_rgb)
    face_res = face.process(frame_rgb)

    # ---------------- Feature Extraction ----------------
    feat: list[float] = []

    # Pose: 33 × (x,y,z)
    if pose_res.pose_landmarks:
        for lm in pose_res.pose_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
        mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        feat.extend([0.0] * 33 * 3)

    # Hands: 0‑2 × 21 landmark
    if hands_res.multi_hand_landmarks:
        for h_lm in hands_res.multi_hand_landmarks:
            for lm in h_lm.landmark:
                feat.extend([lm.x, lm.y, lm.z])
            mp_draw.draw_landmarks(frame, h_lm, mp_hands.HAND_CONNECTIONS)
        if len(hands_res.multi_hand_landmarks) == 1:  # tek el ise diğerini sıfırla
            feat.extend([0.0] * 21 * 3)
    else:
        feat.extend([0.0] * 2 * 21 * 3)

    # FaceMesh: ilk 10 landmark (10 × 3)
    if face_res.multi_face_landmarks:
        for lm in face_res.multi_face_landmarks[0].landmark[:10]:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 10 * 3)

    # Boyut sabitleme
    if len(feat) < FEAT_DIM:
        feat.extend([0.0] * (FEAT_DIM - len(feat)))
    else:
        feat = feat[:FEAT_DIM]

    feat = np.asarray(feat, dtype=np.float32)
    max_val = np.max(np.abs(feat))
    if max_val:
        feat /= max_val

    # ---------------- Prediction Logic ----------------
    if is_predicting:
        frame_buffer.append(feat.tolist())
        if len(frame_buffer) > SEQ_LEN:
            frame_buffer.pop(0)

        # Tahmin zamanı mı?
        if len(frame_buffer) >= 10 and (time.time() - last_pred_time) > PRED_INTERVAL:
            buf = frame_buffer.copy()
            while len(buf) < SEQ_LEN:
                buf.append(buf[-1])
            inp = np.asarray(buf[-SEQ_LEN:]).reshape(1, SEQ_LEN, FEAT_DIM)
            probs = model.predict(inp, verbose=0)
            cls = int(np.argmax(probs))
            pred_history.append(cls)
            last_pred_time = time.time()

        if pred_history:
            most_common = Counter(pred_history[-10:]).most_common(1)[0][0]
            label = label_map.get(most_common, "Bilinmeyen")
            pred_text = f"Tahmin: {label} ({most_common})"
        else:
            pred_text = "Veri toplanıyor…"
    else:
        pred_text = "Tahmin beklemede (B: Başlat, E: Bitir)"

    # ---------------- FPS Hesapla ----------------
    frame_counter += 1
    if (time.time() - fps_start) > 1.0:
        fps = frame_counter / (time.time() - fps_start)
        fps_start = time.time()
        frame_counter = 0

    # ---------------- OSD Çizimleri ----------------
    overlay = frame.copy()
    panel_h = 90
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_h), (20, 20, 20), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Tahmin metni (sol üst)
    frame = draw_text(frame, pred_text, pos=(10, 8), font_size=28, color=(0, 255, 0), bg=False)

    # FPS (sağ üst)
    fps_txt = f"FPS: {fps:.1f}"
    frame = draw_text(frame, fps_txt, pos=(frame.shape[1] - 140, 8), font_size=24,
                      color=(255, 255, 0), bg=False)

    # Kısayol bilgisi (sağ alt)
    keys_txt = "B: Başlat  E: Bitir  Q: Çıkış"
    frame = draw_text(frame, keys_txt, pos=(10, frame.shape[0] - 40), font_size=24,
                      color=(180, 180, 180), bg=True)

    # Son tahminlerden mini geçmiş (sol alt)
    hist_len = 5
    recent = pred_history[-hist_len:] if len(pred_history) >= hist_len else pred_history
    for i, cid in enumerate(recent):
        lbl = label_map.get(cid, "?")
        pos_x = 10 + i * 120
        pos_y = frame.shape[0] - 90
        frame = draw_text(frame, lbl, pos=(pos_x, pos_y), font_size=22,
                          color=(0, 200, 255), bg=True)

    # ---------------- Görüntü Göster ----------------
    scale_percent = 150  # %150 yakınlaştır
    w = int(frame.shape[1] * scale_percent / 100)
    h = int(frame.shape[0] * scale_percent / 100)
    frame_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("İşaret Dili Tanıma", frame_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("b"):
        is_predicting = True
        frame_buffer.clear()
        pred_history.clear()
    elif key == ord("e"):
        is_predicting = False
        frame_buffer.clear()

# ------------------------------------------------------------
# 7. Temizlik
# ------------------------------------------------------------

cap.release()
cv2.destroyAllWindows()
print("Çıkıldı.")
