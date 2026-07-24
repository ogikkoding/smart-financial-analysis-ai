# 💰 Sistem Analisis Transaksi Keuangan Cerdas Menggunakan Machine Learning dan Chatbot Retrieval-Augmented Generation Berbasis Streamlit

## 📖 Deskripsi

Proyek ini bertujuan untuk membangun sistem analisis transaksi keuangan cerdas yang memanfaatkan kombinasi metode **Artificial Intelligence (AI)**, yaitu **Machine Learning** dan **Retrieval-Augmented Generation (RAG)**. Sistem dirancang untuk membantu pengguna menganalisis data transaksi secara otomatis dengan mengklasifikasikan transaksi berdasarkan kategori, jenis transaksi, dan metode pembayaran.

Dalam proses klasifikasi, sistem menggunakan pendekatan **Hybrid IndoBERT-XGBoost**, yaitu **IndoBERT** sebagai *feature extractor* untuk menghasilkan representasi teks transaksi dan **XGBoost** sebagai *classifier* untuk melakukan prediksi. Hasil analisis kemudian disajikan dalam bentuk dashboard interaktif yang memudahkan pengguna memahami pola transaksi.

Selain melakukan analisis dan klasifikasi, sistem juga menyediakan chatbot berbasis **Retrieval-Augmented Generation (RAG)** yang memungkinkan pengguna mengajukan pertanyaan mengenai hasil analisis transaksi. Chatbot memanfaatkan basis pengetahuan yang telah dibangun untuk menghasilkan jawaban yang relevan dan informatif.

Aplikasi dikembangkan menggunakan **Streamlit** sebagai antarmuka web interaktif sehingga pengguna dapat melakukan analisis transaksi, memvisualisasikan hasil, dan berinteraksi dengan chatbot melalui satu platform yang terintegrasi.

---

# 🖼️ Tampilan Aplikasi

### 🏠 Halaman Utama

> Tampilan awal aplikasi sebelum proses analisis transaksi.

![Halaman Utama](images/home.png)

---

### 📂 Upload Dataset

> Pengguna dapat mengunggah dataset transaksi dalam format CSV atau Excel.

![Upload Dataset](images/upload.png)

---

### 📊 Dashboard Analisis

> Dashboard interaktif yang menampilkan hasil analisis transaksi.

![Dashboard](images/dashboard.png)

---

### 🏷️ Hasil Prediksi

> Hasil klasifikasi kategori transaksi, jenis transaksi, dan metode pembayaran.

![Prediksi](images/prediksi.png)

---

### 🤖 Chatbot RAG

> Chatbot berbasis Retrieval-Augmented Generation (RAG) yang dapat menjawab pertanyaan mengenai hasil analisis transaksi.

![Chatbot](images/chatbot.png)

---

# ✨ Fitur

- 📂 Upload dataset transaksi dalam format **CSV** atau **Excel (.xlsx)**.
- 🏷️ Prediksi **kategori transaksi** _(Belanja, Gaji, Hiburan, Investasi, Kesehatan, Makanan & Minuman, Pendidikan, Perjalanan, Transportasi, dan Utilitas)._
- 💰 Prediksi **jenis transaksi** _(Pemasukan dan Pengeluaran)._
- 💳 Prediksi **metode pembayaran** _(BNI, BCA, BRI, Mandiri, QRIS, OVO, GoPay, DANA, ShopeePay, dan Tunai)._
- 📊 Dashboard interaktif untuk visualisasi dan analisis hasil prediksi.
- 📥 Download hasil analisis dan prediksi.
- 🧠 Pembuatan **Knowledge Base** secara otomatis dari data transaksi.
- 🤖 Chatbot berbasis **Retrieval-Augmented Generation (RAG)** untuk menjawab pertanyaan mengenai hasil analisis transaksi.
- 🔍 Pencarian informasi yang relevan menggunakan **FAISS Vector Database**.
- ✨ Integrasi **Google Gemini** untuk menghasilkan respons chatbot yang informatif dan kontekstual.

---

# 🧠 Metode

## Machine Learning

Metode klasifikasi transaksi pada sistem ini menggunakan pendekatan **Hybrid IndoBERT-XGBoost**, dengan tahapan sebagai berikut:

- **IndoBERT Base** sebagai **text encoder** untuk mengubah teks transaksi menjadi representasi vektor (embedding).
- **Mean Pooling** untuk menghasilkan embedding dari keluaran token IndoBERT.
- **XGBoost Classifier** sebagai model klasifikasi yang memanfaatkan embedding hasil encoding untuk melakukan prediksi.

### Model Klasifikasi

Sistem menggunakan tiga model klasifikasi yang dilatih secara terpisah, yaitu:

- 🏷️ **Model Kategori Transaksi**
- 💰 **Model Jenis Transaksi**
- 💳 **Model Metode Pembayaran**

---

# 🤖 Chatbot

Metode chatbot menggunakan:

- Retrieval-Augmented Generation (RAG)
- FAISS Vector Search
- Google Gemini

---

# 🔄 Pipeline Sistem

```text
Dataset
↓
Preprocessing Text
↓
IndoBERT Feature Extraction
↓
Mean Pooling Embedding
↓
XGBoost Classification
↓
Dashboard Analisis
↓
Knowledge Base
↓
FAISS Retrieval
↓
Gemini
↓
RAG Chatbot
```

---

## 🖼️ Diagram Pipeline Sistem

> Diagram alur kerja sistem mulai dari proses klasifikasi hingga chatbot RAG.

![Pipeline Sistem](images/pipeline.png)

---

# 🏗️ Arsitektur Sistem

```text
Dataset
↓
IndoBERT
↓
Embedding
↓
XGBoost
↓
Dashboard
↓
Knowledge Base
↓
FAISS
↓
Gemini
↓
Chatbot
```

---

## 🖼️ Diagram Arsitektur Sistem

> Diagram arsitektur keseluruhan sistem.

![Arsitektur Sistem](images/architecture.png)

---

# 📂 Struktur Folder

```text
project/

├── STREAMLITE KEUANGAN/
│   ├── app.py
│   └── run.bat
│
├── DATASET/
│   └── transaksi_indonesia.xlsx
│
├── MODELS/
│   ├── xgb_model_kategori.json
│   ├── xgb_model_jenis.json
│   ├── xgb_model_pembayaran.json
│   ├── encoder_kategori.npy
│   ├── encoder_jenis.npy
│   ├── encoder_pembayaran.npy
│
├── images/
│   ├── home.png
│   ├── upload.png
│   ├── dashboard.png
│   ├── prediksi.png
│   ├── chatbot.png
│   ├── pipeline.png
│   ├── architecture.png
│   └── output.png
│
├── requirements.txt
│
└── README.md
```

---
# 🛠️ Teknologi

- Python
- Streamlit
- Transformers
- IndoBERT
- XGBoost
- Scikit-Learn
- Pandas
- NumPy
- FAISS
- Plotly
- Google Gemini API

---

# 📊 Output Sistem

Sistem menghasilkan:

- 🏷️ Kategori Transaksi
- 💰 Jenis Transaksi
- 💳 Metode Pembayaran
- 📊 Dashboard Analisis Transaksi
- 📈 Visualisasi Data
- 📥 File Hasil Prediksi (CSV)
- 🤖 Chatbot Berbasis RAG

---

## 🖼️ Contoh Output Sistem

> Contoh hasil analisis transaksi, visualisasi dashboard, dan chatbot.

![Output Sistem](images/output.png)

---

# 📈 Dataset

Dataset yang digunakan merupakan **dataset sintetis** yang dibuat menggunakan ChatGPT untuk keperluan penelitian dan pengembangan sistem analisis transaksi keuangan.

---

## 🖼️ Contoh Dataset

> Contoh data transaksi yang digunakan dalam proses pelatihan dan pengujian model.

![Dataset](images/dataset.png)

---

# 🚀 Cara Menjalankan Aplikasi

## 1. Clone Repository

```bash
git clone https://github.com/ogikkoding/nama-repository.git
```

## 2. Masuk ke Folder Project

```bash
cd nama-repository
```

## 3. Install Seluruh Library

```bash
pip install -r requirements.txt
```

## 4. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

---

# 💻 Platform Pengembangan

- Google Colab (Pelatihan Model)
- Google Drive (Penyimpanan Model)
- Visual Studio Code (Pengembangan Aplikasi)
- Streamlit (Implementasi Aplikasi)

---

## 👨‍💻 Author

**Yogi Irawan**

- 🎓 Undergraduate Student of Informatics
- 🤖 Interested in Artificial Intelligence & Computer Vision
- 📧 Email: yogiirawan490@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/yogi-irawan-ab146a387
- 🐙 GitHub: https://github.com/ogikkoding

---

# 📄 License

This project is developed for academic research and educational purposes.