# Türk İşaret Dili (TİD) Tanıma Sistemi  

Bu proje, **Türk İşaret Dili (TİD)** işaretlerini otomatik olarak tanıyan bir sistem geliştirmeyi amaçlamaktadır.  
Sistem, kullanıcıların işaret diliyle verdiği girdileri **metin** ve **ses** çıktısına dönüştürerek iletişimde köprü kurmayı hedefler.  

---

## 📂 Veri Seti  
Proje, **[AUTSL (A Large Scale Turkish Sign Language Dataset)](https://cvml.ankara.edu.tr/datasets/)** veri seti kullanılarak geliştirilmiştir.  
AUTSL, farklı katılımcılar tarafından yapılmış binlerce Türk İşaret Dili videosunu içermektedir.  

---

## ⚙️ Yöntem  

- **Ön işleme:** Video verileri 30 frame uzunluğa normalize edilip `.npy` formatına dönüştürülmüştür.  
- **Model:** BiLSTM / CNN tabanlı derin öğrenme modelleri kullanılmıştır.  
- **Gerçek zamanlı işaret yakalama:** MediaPipe ile el, yüz ve vücut landmark bilgileri çıkarılmıştır.  
- **Tahmin:** Eğitilen model landmark verilerini işleyerek işaretin hangi sınıfa ait olduğunu belirler.  
- **Mobil uygulama:** TensorFlow Lite entegrasyonu ile Android ortamında gerçek zamanlı işaret dili tanıma sağlanmıştır.  

---

## 📌 Scriptler  

| Script | Açıklama |
|--------|----------|
| `extract_landmarks.py` | AUTSL videolarından landmark verilerini çıkarır ve `.npy` dosyalarına kaydeder. |
| `filter_first_100_classes.py` | Veri setinden seçilen ilk 100 sınıfa ait videoları ayıklar ve kopyalar. |
| `csv_filter_100_classes.py` | CSV dosyasındaki verilerden alfabetik sıraya göre ilk 100 sınıfı filtreler ve ilgili videoları kopyalar. |
| `convert_to_tflite.py` | Eğitilmiş Keras modelini TensorFlow Lite formatına dönüştürür. |
| `data_augmentation.py` | Eğitim verilerini çeşitlendirmek için veri artırma (data augmentation) uygular. |
| `train_sign_model.py` | Landmark verileriyle BiLSTM/CNN tabanlı modeli eğitir, test eder ve performansını değerlendirir. |
| `compare_models.py` | Farklı derin öğrenme modellerini eğitir ve performanslarını karşılaştırır. |
| `realtime_sign_gui.py` | Kamera üzerinden landmark çıkarıp Keras modeliyle gerçek zamanlı TİD tahmini yapan GUI. |
| `realtime_sign_tflite.py` | Kamera üzerinden TFLite modeliyle gerçek zamanlı TİD tanıma uygulaması. |
