import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, GRU, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- Veri yükleme fonksiyonunu aynen kullan ---
def load_data(split, num_frames=30):
    EXPECTED_FEATURES = 225
    X, y, skipped = [], [], []
    dirs = [f"first_100/processed_data_100/{split}/"]

    def extract_label(file):
        match = re.search(r'_(\d+)(?:_aug\d+)?\.npy$', file)
        if match:
            return int(match.group(1))
        return None

    def process_file(file, data_dir):
        try:
            path = os.path.join(data_dir, file)
            data = np.load(path)
            if data.ndim != 2 or data.shape[1] != EXPECTED_FEATURES:
                return None
            if data.shape[0] > num_frames:
                data = data[:num_frames]
            elif data.shape[0] < num_frames:
                padding = np.zeros((num_frames - data.shape[0], EXPECTED_FEATURES))
                data = np.vstack((data, padding))
            label = extract_label(file)
            if label is None:
                return None
            return data, label
        except:
            return None

    for data_dir in dirs:
        files = os.listdir(data_dir)
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda f: process_file(f, data_dir), files), total=len(files), desc=f"{split} verileri yükleniyor"))
            for result in results:
                if result is not None:
                    data, label = result
                    X.append(data)
                    y.append(label)
                else:
                    skipped.append(1)

    print(f"{split} kümesinde atlanan dosya sayısı: {len(skipped)}")
    return np.array(X), np.array(y)

# --- Veri Yükle ---
X_train, y_train = load_data("train")
X_val, y_val = load_data("val")
X_test, y_test = load_data("test")

# --- Verileri birleştir ve normalizasyon ---
X_full = np.concatenate([X_train, X_val], axis=0)
y_full = np.concatenate([y_train, y_val], axis=0)
X_full = X_full / np.max(X_full)
X_test = X_test / np.max(X_test)

print(f"Eğitim verisi şekli: {X_full.shape}")
print(f"Test verisi şekli: {X_test.shape}")

# --- Etiketleme ---
le = LabelEncoder()
y_encoded = le.fit_transform(y_full)
y_cat = to_categorical(y_encoded)
y_test_cat = to_categorical(le.transform(y_test))

num_classes = y_cat.shape[1]
print(f"Sınıf sayısı: {num_classes}")

# --- Eğitim/Doğrulama böl ---
X_train, X_val, y_train, y_val = train_test_split(X_full, y_cat, test_size=0.2, stratify=y_cat, random_state=42)

# --- Class weights ---
y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights))

# --- Model tanımları ---

def build_cnn(input_shape=(30,225), num_classes=num_classes):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_shape=(30,225), num_classes=num_classes):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_lstm(input_shape=(30,225), num_classes=num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(30, 225)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),

        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_gru(input_shape=(30,225), num_classes=num_classes):
    model = Sequential([
        GRU(128, input_shape=input_shape, return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Eğitim ve değerlendirme fonksiyonu ---
def train_and_evaluate(model_builder, model_name):
    print(f"\n--- {model_name} modeli eğitiliyor ---")
    model = model_builder()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f"best_{model_name}.keras", monitor="val_loss", save_best_only=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop, checkpoint],
        shuffle=True
    )

    print(f"\n--- {model_name} modeli test setinde değerlendiriliyor ---")
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test_cat, axis=1)

    print(classification_report(y_true_labels, y_pred_labels, target_names=[str(cls) for cls in le.classes_]))
    
    return history, classification_report(y_true_labels, y_pred_labels, target_names=[str(cls) for cls in le.classes_], output_dict=True)

# --- Modelleri sırayla çalıştır ve sonuçları topla ---
models = {
    "CNN": build_cnn,
    "LSTM": build_lstm,
    "CNN+LSTM": build_cnn_lstm,
    "GRU": build_gru
}

histories = {}
reports = {}

for name, builder in models.items():
    history, report = train_and_evaluate(builder, name)
    histories[name] = history
    reports[name] = report

# --- Eğitim grafikleri karşılaştırma ---
plt.figure(figsize=(16, 8))
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f'{name} Val Accuracy')
plt.title('Modellerin Doğrulama Doğruluğu Karşılaştırması')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

