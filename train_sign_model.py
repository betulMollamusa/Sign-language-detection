import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, BatchNormalization, Dropout, Bidirectional, LSTM, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.utils import to_categorical

# ------------------------------------------------------------
# 1. Veri Yükleme Fonksiyonu
# ------------------------------------------------------------

def load_data(split: str, num_frames: int = 30, expected_features: int = 225):
    X, y, skipped = [], [], []
    data_dir = f"first_100/processed_data_100/{split}/"

    def extract_label(filename: str):
        m = re.search(r"_(\d+)(?:_aug\d+)?\.npy$", filename)
        return int(m.group(1)) if m else None

    def process_file(fname: str):
        try:
            arr = np.load(os.path.join(data_dir, fname))
            if arr.ndim != 2 or arr.shape[1] != expected_features:
                return None
            max_per_frame = arr.max(axis=1, keepdims=True)
            max_per_frame[max_per_frame == 0] = 1
            arr = arr / max_per_frame
            if arr.shape[0] > num_frames:
                arr = arr[:num_frames]
            elif arr.shape[0] < num_frames:
                pad = np.repeat(arr[-1:], num_frames - arr.shape[0], axis=0)
                arr = np.vstack((arr, pad))
            label = extract_label(fname)
            return (arr, label) if label is not None else None
        except Exception:
            return None

    files = os.listdir(data_dir)
    with ThreadPoolExecutor() as ex:
        for result in tqdm(ex.map(process_file, files), total=len(files), desc=f"{split} verileri yükleniyor"):
            if result is None:
                skipped.append(1)
            else:
                data, lbl = result
                X.append(data)
                y.append(lbl)

    print(f"{split} kümesinde atlanan dosya sayısı: {len(skipped)}")
    return np.array(X, dtype=np.float32), np.array(y)

X_train_raw, y_train_raw = load_data("train")
X_val_raw,   y_val_raw   = load_data("val")
X_test,      y_test      = load_data("test")

X_full = np.concatenate([X_train_raw, X_val_raw], axis=0)
y_full = np.concatenate([y_train_raw, y_val_raw], axis=0)

print(f"\nEğitim verisi şekli: {X_full.shape}")
print(f"Test   verisi şekli: {X_test.shape}\n")

le = LabelEncoder()
y_encoded = le.fit_transform(y_full)
num_classes = len(le.classes_)

y_cat       = to_categorical(y_encoded, num_classes)
y_test_cat  = to_categorical(le.transform(y_test), num_classes)

print(f"Sınıf sayısı: {num_classes}\n")

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_cat, test_size=0.2, stratify=y_cat, random_state=42)

class_weights = compute_class_weight(
    class_weight='balanced', classes=np.arange(num_classes),
    y=np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(class_weights))

model = Sequential([
    Conv1D(64,  3, padding='same', activation='relu', input_shape=(30, 225)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.25),

    Conv1D(256, 3, padding='same', activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),

    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint('newest_model.keras', monitor='val_loss', save_best_only=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

print("Eğitim başlıyor...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint, reduce_lr],
    shuffle=True,
    verbose=1
)

def plot_history(h):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(h.history['loss'], label='Eğitim Kaybı')
    plt.plot(h.history['val_loss'], label='Doğrulama Kaybı')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Kayıp')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(h.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(h.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Doğruluk')
    plt.legend()

    plt.tight_layout(); plt.show()

plot_history(history)

print("\n🧪 Baseline değerlendirme (augment yok)...")
start = time.time()
baseline_proba  = model.predict(X_test, verbose=0)
end = time.time()
baseline_pred   = np.argmax(baseline_proba, axis=1)
y_true_labels   = np.argmax(y_test_cat, axis=1)
print(f"Velocity (no TTA): {(end - start) / len(X_test):.4f} s/sample")
print(f"Baseline Accuracy : {accuracy_score(y_true_labels, baseline_pred):.4f}\n")
print("Classification Report (Baseline):\n")
print(classification_report(y_true_labels, baseline_pred, target_names=[str(c) for c in le.classes_]))

SEQ_LEN = 30
FEAT_DIM = 225

def add_noise(sample, sigma=0.02):
    return sample + np.random.normal(0, sigma, sample.shape)

def scale_range(sample, alpha_range=(0.95, 1.05)):
    alpha = np.random.uniform(*alpha_range)
    return sample * alpha

def time_shift(sample, max_shift=2):
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return sample
    if shift > 0:
        pad = np.repeat(sample[:1], shift, axis=0)
        return np.concatenate([pad, sample[:-shift]], axis=0)
    pad = np.repeat(sample[-1:], -shift, axis=0)
    return np.concatenate([sample[-shift:], pad], axis=0)

def time_mask(sample, p=0.1):
    mask_len = int(p * SEQ_LEN)
    idx = np.random.choice(SEQ_LEN, mask_len, replace=False)
    sample[idx] = sample[-1]
    return sample

AUG_FNS = [add_noise, scale_range, time_shift, time_mask]

def make_augmentations(sample, K=4):
    aug_samples = [sample]
    for _ in range(K):
        aug = sample.copy()
        for fn in np.random.choice(AUG_FNS, np.random.randint(1, 4), replace=False):
            aug = fn(aug)
        aug_samples.append(aug)
    return np.stack(aug_samples)

def predict_tta(model, X, K=4):
    num_classes = model.output_shape[-1]
    N = X.shape[0]
    probs_sum = np.zeros((N, num_classes), dtype=np.float32)
    for i in range(N):
        aug_batch = make_augmentations(X[i], K)
        preds = model.predict(aug_batch, verbose=0)
        probs_sum[i] = preds.mean(axis=0)
    return probs_sum

print("\nTest Time Augmentation (K=4) değerlendirme...")
start = time.time()
proba_tta = predict_tta(model, X_test, K=4)
end = time.time()
pred_tta  = np.argmax(proba_tta, axis=1)
print(f"Velocity (TTA, K=4): {(end - start) / len(X_test):.4f} s/sample")
print(f"TTA Accuracy      : {accuracy_score(y_true_labels, pred_tta):.4f}\n")
print("Classification Report (TTA):\n")
print(classification_report(y_true_labels, pred_tta, target_names=[str(c) for c in le.classes_]))
