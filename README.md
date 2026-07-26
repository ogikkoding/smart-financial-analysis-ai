# 💰 Intelligent Financial Transaction Analysis System Using Machine Learning and Retrieval-Augmented Generation Chatbot Based on Streamlit

## 📖 Description

This project aims to develop an intelligent financial transaction analysis system by integrating **Artificial Intelligence (AI)** techniques, specifically **Machine Learning** and **Retrieval-Augmented Generation (RAG)**. The system is designed to assist users in automatically analyzing financial transaction data by classifying transactions based on transaction category, transaction type, and payment method.

The classification process employs a **Hybrid IndoBERT-XGBoost** approach, where **IndoBERT** serves as a _feature extractor_ to generate semantic representations of transaction texts, while **XGBoost** acts as the _classifier_ to perform prediction. The classification results are presented through an interactive dashboard, enabling users to gain insights into their transaction patterns more effectively.

In addition to transaction classification and analysis, the system provides a **Retrieval-Augmented Generation (RAG)** chatbot that allows users to ask questions related to the analysis results. The chatbot utilizes a knowledge base generated from transaction data to produce accurate, relevant, and context-aware responses.

The application is developed using **Streamlit** as an interactive web interface, allowing users to upload transaction datasets, analyze financial records, visualize results, and interact with the AI chatbot through a single integrated platform.

---

# 🖼️ Application Preview

### 🏠 Home Page

> The main interface displayed before the transaction analysis process begins.

![Home Page](IMAGES/home.png)

---

### 📊 Analysis Dashboard and Prediction Results

> Interactive dashboard presenting transaction analysis and visualization.

![Dashboard](IMAGES/analisis.png)

---

### 🤖 RAG Chatbot

> Retrieval-Augmented Generation (RAG) chatbot that answers questions related to transaction analysis results.

![Chatbot](images/chatboot.png)

---

# ✨ Features

- 📂 Upload transaction datasets in **CSV** or **Excel (.xlsx)** format.
- 🏷️ Predict **transaction categories** _(Shopping, Salary, Entertainment, Investment, Healthcare, Food & Beverage, Education, Travel, Transportation, and Utilities)._
- 💰 Predict **transaction types** _(Income and Expense)._
- 💳 Predict **payment methods** _(BNI, BCA, BRI, Mandiri, QRIS, OVO, GoPay, DANA, ShopeePay, and Cash)._
- 📊 Interactive dashboard for transaction analysis and visualization.
- 📥 Download prediction and analysis results.
- 🧠 Automatically generate a **Knowledge Base** from transaction data.
- 🤖 AI chatbot powered by **Retrieval-Augmented Generation (RAG)**.
- 🔍 Retrieve relevant information using **FAISS Vector Database**.
- ✨ Integrate **Google Gemini** to generate informative and context-aware chatbot responses.

---

# 🧠 Methodology

## Machine Learning

The transaction classification module utilizes a **Hybrid IndoBERT-XGBoost** approach consisting of the following stages:

- **IndoBERT Base** as the **text encoder** to transform transaction descriptions into dense vector representations (embeddings).
- **Mean Pooling** to generate sentence-level embeddings from IndoBERT token outputs.
- **XGBoost Classifier** to classify transactions based on the generated embeddings.

### Classification Models

The system employs three independently trained classification models:

- 🏷️ **Transaction Category Model**
- 💰 **Transaction Type Model**
- 💳 **Payment Method Model**

---

# 🤖 Chatbot

The chatbot module is built using:

- Retrieval-Augmented Generation (RAG)
- FAISS Vector Search
- Google Gemini

---

# 🔄 System Pipeline

```text
Dataset
↓
Text Preprocessing
↓
IndoBERT Feature Extraction
↓
Mean Pooling Embedding
↓
XGBoost Classification
↓
Analysis Dashboard
↓
Knowledge Base
↓
FAISS Retrieval
↓
Google Gemini
↓
RAG Chatbot
```

---

## 🖼️ System Pipeline Diagram

> Overview of the system workflow from transaction classification to the RAG chatbot.

![System Pipeline](images/pipeline.png)

---

# 🏗️ System Architecture

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
Google Gemini
↓
Chatbot
```

---

## 🖼️ System Architecture Diagram

> Overall architecture of the intelligent financial transaction analysis system.

![System Architecture](images/architecture.png)

---

# 📂 Project Structure

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

# 🛠️ Technologies Used

- Python
- Streamlit
- Transformers
- IndoBERT
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- FAISS
- Plotly
- Google Gemini API

---

# 📊 System Output

The application provides:

- 🏷️ Predicted Transaction Category
- 💰 Predicted Transaction Type
- 💳 Predicted Payment Method
- 📊 Interactive Transaction Analysis Dashboard
- 📈 Data Visualization
- 📥 Downloadable Prediction Results (CSV)
- 🤖 AI-Powered RAG Chatbot

---

## 🖼️ Sample Output

> Example of transaction analysis results, dashboard visualization, and chatbot responses.

![System Output](images/output.png)

---

# 📈 Dataset

The dataset used in this project is a **synthetic financial transaction dataset** generated using **ChatGPT** for research and educational purposes in developing an intelligent financial transaction analysis system.

---

## 🖼️ Sample Dataset

> Sample transaction records used for model training and evaluation.

![Dataset](images/dataset.png)

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/ogikkoding/nama-repository.git
```

## 2. Navigate to the Project Directory

```bash
cd nama-repository
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run app.py
```

---

# 💻 Development Environment

- Google Colab (Model Training)
- Google Drive (Model Storage)
- Visual Studio Code (Application Development)
- Streamlit (Application Deployment)

---

# 👨‍💻 Author

**Yogi Irawan**

- 🎓 Bachelor's Student in Informatics
- 🤖 Research Interests: Artificial Intelligence, Natural Language Processing, Machine Learning, and Financial Data Analytics
- 📧 Email: yogiirawan490@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/yogi-irawan-ab146a387
- 🐙 GitHub: https://github.com/ogikkoding

---

# 📄 License

This project is licensed under the **MIT License**.

Copyright (c) 2026 **Yogi Irawan**
