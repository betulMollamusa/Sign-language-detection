import numpy as np
import os
from tqdm import tqdm
import random

# Parametreler
DATA_DIR = 'first_100/processed_data_100/train'  # Hem okuma hem yazma için aynı klasör
AUG_PER_SAMPLE = 2
EXPECTED_FEATURES = 225
NUM_FRAMES = 30

def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def time_shift(data, shift_range=3):
    shift = random.randint(-shift_range, shift_range)
    return np.roll(data, shift, axis=0)

# Tüm orijinal dosyaları listele (augment dosyaları hariç)
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy') and '_aug' not in f]

for file in tqdm(files, desc="Veri artırılıyor"):
    path = os.path.join(DATA_DIR, file)

    try:
        data = np.load(path)
    except Exception as e:
        print(f"{file} yüklenemedi, atlanıyor. Hata: {e}")
        continue

    # Boyut kontrolü
    if data.ndim != 2 or data.shape[1] != EXPECTED_FEATURES:
        print(f"{file} beklenen shape değil, atlanıyor. Gerçek boyut: {data.shape}")
        continue

    for i in range(AUG_PER_SAMPLE):
        try:
            augmented = add_noise(data)
            augmented = time_shift(augmented)

            # Frame sayısını ayarla
            if augmented.shape[0] > NUM_FRAMES:
                augmented = augmented[:NUM_FRAMES]
            elif augmented.shape[0] < NUM_FRAMES:
                pad = np.zeros((NUM_FRAMES - augmented.shape[0], EXPECTED_FEATURES))
                augmented = np.vstack((augmented, pad))

            # Yeni dosya ismi: orijinal isim + _aug{i}.npy
            new_filename = file.replace('.npy', f'_aug{i}.npy')
            new_path = os.path.join(DATA_DIR, new_filename)
            np.save(new_path, augmented)

        except Exception as e:
            print(f"Augmentasyon hatası ({file}): {e}")

print("✅ Veri artırma tamamlandı. Tüm dosyalar aynı klasöre yazıldı.")
