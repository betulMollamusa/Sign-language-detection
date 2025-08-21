import os
import shutil
import pandas as pd

# Klasör yolları
train_dir = "test"
output_dir = "first_100/test_first_100_classes"
label_file = "first_100/test_labels_first_100.csv"

# Çıkış klasörünü oluştur
os.makedirs(output_dir, exist_ok=True)

# CSV'yi başlık olmadan oku
df = pd.read_csv(label_file, header=None, names=["filename", "label"])

# İlk 100 sınıfı al
first_100_classes = sorted(df['label'].unique())[:100]

# İlk 100 sınıfa ait satırları filtrele
filtered_df = df[df['label'].isin(first_100_classes)]

# .mp4 dosya adını oluştur ('_color.mp4' ekle)
filtered_df['filename_mp4'] = filtered_df['filename'] + "_color.mp4"

# Dosyaları kopyala
for _, row in filtered_df.iterrows():
    src = os.path.join(train_dir, row['filename_mp4'])
    dst = os.path.join(output_dir, row['filename_mp4'])
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Dosya bulunamadı: {src}")
