# Sistem Deteksi XSS Menggunakan Machine Learning

Program Python komprehensif untuk mendeteksi serangan Cross-Site Scripting (XSS) menggunakan model machine learning XGBoost dan LightGBM dengan SMOTE untuk menangani data yang tidak seimbang.

## 🎯 Gambaran Umum

Proyek ini mengimplementasikan sistem deteksi XSS yang robust dengan membandingkan performa dua model machine learning yang powerful:
- **XGBoost**: Framework gradient boosting
- **LightGBM**: Framework gradient boosting yang dioptimalkan untuk kecepatan dan efisiensi memori

Sistem ini menggunakan SMOTE (Synthetic Minority Oversampling Technique) untuk menangani dataset yang tidak seimbang dan mencakup feature engineering komprehensif yang dirancang khusus untuk deteksi XSS.

## 📊 Dataset

Sistem ini dirancang untuk bekerja dengan **Dataset Cross-Site Scripting (XSS) untuk Deep Learning** dari Kaggle:
- **URL**: https://www.kaggle.com/datasets/syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning
- **Format**: CSV dengan sampel teks dan label biner (XSS vs Normal)

## 🚀 Fitur Utama

### Model Machine Learning
- **XGBoost Classifier**: Gradient boosting berperforma tinggi
- **LightGBM Classifier**: Gradient boosting yang cepat dan efisien memori
- **Random Forest**: Baseline ensemble learning
- **Integrasi SMOTE**: Menangani data yang tidak seimbang secara efektif

### Feature Engineering
- **Fitur Manual**: Pola khusus XSS, tag HTML, karakter khusus
- **Fitur Teks**: Vektorisasi TF-IDF dengan n-gram
- **Fitur Statistik**: Panjang teks, jumlah kata, entropi
- **Fitur Keamanan**: Kata kunci mencurigakan, pola URL, alamat IP

### Analisis Komprehensif
- **Analisis Data Eksploratori**: Distribusi data, keseimbangan kelas, statistik teks
- **Perbandingan Model**: Metrik performa di semua model
- **Visualisasi**: Kurva ROC, confusion matrix, feature importance
- **Hyperparameter Tuning**: Optimasi grid search

## 📁 Struktur Proyek

```
xss_ml/
├── requirements.txt          # Dependensi Python
├── xss_detection.py         # Sistem deteksi XSS utama
├── data_loader.py           # Pemuatan dan preprocessing data
├── run_xss_detection.py     # Contoh workflow lengkap
├── test_installation.py     # Tes validasi sistem
├── README.md                # File ini
└── File yang Dihasilkan:
    ├── xss_detection_models.pkl    # Model terlatih
    ├── class_distribution.png      # Visualisasi data
    ├── model_evaluation.png        # Perbandingan model
    ├── xss_data_analysis.png       # Plot analisis data
    ├── xss_wordclouds.png          # Word cloud
    └── xss_detection_report.txt    # Laporan ringkasan
```

## 🛠️ Instalasi

1. **Clone repository**:
```bash
git clone <repository-url>
cd xss_ml
```

2. **Install dependensi**:
```bash
# Buat virtual environment
python3 -m venv xss_env
source xss_env/bin/activate

# Install paket yang diperlukan
pip install -r requirements.txt
```

3. **Download data NLTK** (otomatis saat pertama kali dijalankan):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🎮 Cara Penggunaan

### Memulai Cepat

Jalankan pipeline deteksi XSS lengkap:
```bash
source xss_env/bin/activate
python3 run_xss_detection.py
```

### Menggunakan Dataset Sendiri

1. **Download dataset Kaggle**
2. **Update path data** di `run_xss_detection.py`:
```python
df = data_loader.load_kaggle_dataset('path_ke_dataset_anda.csv')
```

### Komponen Individual

**Pemuatan dan Eksplorasi Data**:
```python
from data_loader import XSSDataLoader

loader = XSSDataLoader()
df = loader.load_kaggle_dataset('dataset.csv')
loader.explore_dataset(df)
```

**Deteksi XSS**:
```python
from xss_detection import XSSDetector

detector = XSSDetector()
X, y = detector.preprocess_data(df)
results = detector.train_models(X, y)
detector.evaluate_models(results)
```

**Membuat Prediksi**:
```python
# Muat model terlatih
detector.load_models('xss_detection_models.pkl')

# Prediksi XSS
text = '<script>alert("XSS")</script>'
prediction = detector.predict(text, 'XGBoost')
print(f"Prediksi: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

## 📈 Performa Model

Sistem mengevaluasi model menggunakan berbagai metrik:

- **Accuracy**: Ketepatan keseluruhan
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Rata-rata harmonik dari precision dan recall
- **ROC AUC**: Area di bawah kurva ROC

### Performa yang Diharapkan
Dengan dataset sampel, rentang performa tipikal:
- **XGBoost**: F1-Score ~0.95-0.98
- **LightGBM**: F1-Score ~0.94-0.97
- **Random Forest**: F1-Score ~0.92-0.95

## 🔧 Konfigurasi

### Parameter Model

**XGBoost**:
```python
xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

**LightGBM**:
```python
lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

### Feature Engineering

**Parameter TF-IDF**:
```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
)
```

**Parameter SMOTE**:
```python
SMOTE(random_state=42)
```

## 📊 Visualisasi

Sistem menghasilkan beberapa visualisasi:

1. **Distribusi Kelas**: Diagram pie menunjukkan sampel XSS vs Normal
2. **Statistik Teks**: Distribusi panjang dan jumlah kata
3. **Perbandingan Model**: Visualisasi metrik performa
4. **Kurva ROC**: Visualisasi performa model
5. **Confusion Matrix**: Breakdown akurasi klasifikasi
6. **Feature Importance**: Fitur terpenting untuk deteksi
7. **Word Cloud**: Representasi visual teks XSS dan normal

## 🧪 Pengujian

### Contoh Pola XSS yang Terdeteksi
- `<script>alert("XSS")</script>`
- `javascript:alert(1)`
- `<img src=x onerror=alert(1)>`
- `<iframe src="javascript:alert(1)"></iframe>`
- `eval(String.fromCharCode(...))`
- `document.cookie`
- `window.location`

### Mode Pengujian Interaktif
Sistem mencakup mode interaktif untuk pengujian real-time:
```bash
python3 run_xss_detection.py
# Pilih 'y' saat diminta untuk mode interaktif
```

## 🔍 Detail Fitur

### Fitur Manual (40+ fitur)
- **Pola XSS**: Tag script, protokol JavaScript, event handler
- **Elemen HTML**: Tag berbahaya seperti iframe, object, embed
- **Karakter Khusus**: Kurung sudut, tanda kutip, tanda kurung
- **Kata Kunci Mencurigakan**: alert, eval, document, window
- **Statistik**: Entropi teks, jumlah URL, alamat IP

### Fitur TF-IDF (5000 fitur)
- **N-gram**: Kombinasi 1-gram, 2-gram, 3-gram
- **Stop Words**: Dihapus untuk sinyal yang lebih baik
- **Filter Frekuensi**: Ambang batas frekuensi dokumen min/max

## 🚨 Pertimbangan Keamanan

1. **False Positive**: Mungkin menandai potongan kode yang sah
2. **Evasion**: Serangan canggih mungkin melewati deteksi
3. **Update**: Perlu pelatihan ulang reguler untuk pola serangan baru
4. **Performa**: Pertimbangan deployment real-time

## 🤝 Kontribusi

1. Fork repository
2. Buat branch fitur
3. Lakukan perubahan
4. Tambahkan tes jika berlaku
5. Submit pull request

## 📝 Lisensi

Proyek ini open source dan tersedia di bawah MIT License.

## 🙏 Penghargaan

- **Dataset Kaggle**: Cross-Site Scripting (XSS) Dataset for Deep Learning
- **XGBoost**: Framework Extreme Gradient Boosting
- **LightGBM**: Light Gradient Boosting Machine
- **scikit-learn**: Library machine learning
- **SMOTE**: Synthetic Minority Oversampling Technique

## 📞 Dukungan

Untuk pertanyaan atau masalah:
1. Periksa file `xss_detection_report.txt` yang dihasilkan untuk hasil detail
2. Tinjau file visualisasi untuk wawasan
3. Periksa output konsol untuk informasi debugging

## 🔄 Peningkatan Masa Depan

- Model deep learning (LSTM, BERT)
- Deployment API web real-time
- Feature engineering lanjutan
- Metode ensemble
- Dukungan multi-bahasa
- Optimasi performa 
