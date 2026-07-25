# ==========================================================
# IMPORT
# ==========================================================
import os
import re
import faiss
import torch
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel

# ==========================================================
# KONFIGURASI STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="Sistem Analisis Transaksi Keuangan Cerdas",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💰 Sistem Analisis Transaksi Keuangan Cerdas")
st.caption(
    "Hybrid IndoBERT + XGBoost Classification "
    "dan Retrieval-Augmented Generation (RAG) Chatbot"
)


# ==========================================================
# KONFIGURASI GEMINI
# ==========================================================

from dotenv import load_dotenv
import os
import google.generativeai as genai

# Membaca file .env
load_dotenv()

# Mengambil API Key dari .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Konfigurasi Gemini
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel(
    model_name="gemini-3.5-flash"
)

# ==========================================================
# LOAD MODEL INDOBERT
# ==========================================================
INDOBERT_MODEL_NAME = "indobenchmark/indobert-base-p1"
@st.cache_resource
def load_indobert():
    """
    Load tokenizer dan model IndoBERT.
    Model hanya dimuat sekali selama aplikasi berjalan.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        INDOBERT_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        INDOBERT_MODEL_NAME
    )

    model.eval()

    return tokenizer, model


tokenizer, indobert_model = load_indobert()


# ==========================================================
# LOAD MODEL XGBOOST
# ==========================================================
@st.cache_resource
def load_xgboost_models():
    """
    Memuat seluruh model XGBoost untuk inferensi.
    """

    model_kategori = xgb.XGBClassifier()
    model_kategori.load_model(
        "../MODELS/xgb_model_kategori.json"
    )

    model_jenis = xgb.XGBClassifier()
    model_jenis.load_model(
        "../MODELS/xgb_model_jenis.json"
    )

    model_pembayaran = xgb.XGBClassifier()
    model_pembayaran.load_model(
        "../MODELS/xgb_model_pembayaran.json"
    )

    return {
        "kategori": model_kategori,
        "jenis": model_jenis,
        "pembayaran": model_pembayaran,
    }
xgb_models = load_xgboost_models()


# ==========================================================
# LOAD ENCODER
# ==========================================================
@st.cache_resource
def load_encoders():
    """
    Memuat label encoder hasil training.
    """
    encoders = {
        "kategori": np.load(
            "../MODELS/encoder_kategori.npy",
            allow_pickle=True
        ),

        "jenis": np.load(
            "../MODELS/encoder_jenis.npy",
            allow_pickle=True
        ),

        "pembayaran": np.load(
            "../MODELS/encoder_pembayaran.npy",
            allow_pickle=True
        ),
    }
    return encoders

# ============================
# BARIS INI HARUS ADA
# ============================
encoders = load_encoders()

# ==========================================================
# UTILITY FUNCTION
# ==========================================================
def read_uploaded_file(uploaded_file):
    """
    Membaca file CSV atau Excel.
    """

    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)

    else:
        raise ValueError(
            "Format file tidak didukung."
        )
    return df


def validate_dataset(df):
    """
    Menentukan kolom teks yang akan digunakan
    sebagai input model.
    """
    if df is None:
        raise ValueError("Dataset kosong.")

    if df.empty:
        raise ValueError("Dataset tidak memiliki data.")

    object_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if len(object_columns) == 0:
        raise ValueError(
            "Tidak ditemukan kolom bertipe teks."
        )

    priority_columns = [
        "deskripsi",
        "keterangan",
        "transaksi",
        "uraian",
        "description"
    ]

    object_lower = {
        col.lower(): col
        for col in object_columns
    }

    for col in priority_columns:
        if col in object_lower:
            return object_lower[col]

    return object_columns[0]


def safe_text(text):
    """
    Mengubah seluruh nilai menjadi string.
    """

    if pd.isna(text):
        return ""

    return str(text)


# ==========================================================
# TEXT PREPROCESSING
# ==========================================================
def preprocess_text(text):
    """
    Membersihkan teks transaksi.
    """

    text = safe_text(text)

    text = text.lower()

    text = re.sub(
        r"http\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# MEAN POOLING
# ==========================================================
def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling dari output IndoBERT.
    """

    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    pooled = torch.sum(
        token_embeddings * input_mask_expanded,
        dim=1
    )

    pooled = pooled / torch.clamp(
        input_mask_expanded.sum(dim=1),
        min=1e-9
    )

    return pooled


# ==========================================================
# BATCH EMBEDDING
# ==========================================================
def batch_embedding(
    texts,
    tokenizer,
    model,
    batch_size=32,
    max_length=128
):
    """
    Menghasilkan embedding IndoBERT
    secara batch agar proses inferensi
    lebih cepat.
    """

    all_embeddings = []

    texts = [
        preprocess_text(text)
        for text in texts
    ]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    with torch.no_grad():

        for start in range(
            0,
            len(texts),
            batch_size
        ):

            batch = texts[
                start:start + batch_size
            ]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            output = model(**encoded)

            embedding = mean_pooling(
                output,
                encoded["attention_mask"]
            )

            embedding = (
                embedding
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            all_embeddings.append(
                embedding
            )

    embeddings = np.vstack(all_embeddings)

    return embeddings

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================
def decode_prediction(predictions, encoder):
    """
    Mengubah hasil prediksi numerik menjadi label asli.
    """
    predictions = np.asarray(predictions).astype(int)
    return encoder[predictions]


def predict_labels(embeddings):
    """
    Melakukan prediksi menggunakan seluruh model XGBoost.
    """
    predictions = {}

    # -----------------------------
    # Kategori
    # -----------------------------
    pred_kategori = xgb_models["kategori"].predict(
        embeddings
    )

    predictions["kategori"] = decode_prediction(
        pred_kategori,
        encoders["kategori"]
    )

    # -----------------------------
    # Jenis
    # -----------------------------
    pred_jenis = xgb_models["jenis"].predict(
        embeddings
    )

    predictions["jenis"] = decode_prediction(
        pred_jenis,
        encoders["jenis"]
    )

    # -----------------------------
    # Pembayaran
    # -----------------------------
    pred_pembayaran = xgb_models["pembayaran"].predict(
        embeddings
    )

    predictions["pembayaran"] = decode_prediction(
        pred_pembayaran,
        encoders["pembayaran"]
    )
    return predictions


def predict_dataframe(
    df,
    text_column,
    tokenizer,
    model
):
    """
    Pipeline inferensi lengkap.
    """

    result_df = df.copy()

    texts = (
        result_df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = batch_embedding(
        texts=texts,
        tokenizer=tokenizer,
        model=model
    )

    predictions = predict_labels(
        embeddings
    )

    result_df["Kategori"] = (
        predictions["kategori"]
    )

    result_df["Jenis"] = (
        predictions["jenis"]
    )

    result_df["Pembayaran"] = (
        predictions["pembayaran"]
    )
    return result_df, embeddings

# ==========================================================
# DASHBOARD FUNCTION
# ==========================================================
def detect_amount_column(df):
    """
    Mendeteksi kolom nominal secara otomatis.
    """

    priority_columns = [
        "nominal",
        "jumlah",
        "amount",
        "nilai",
        "total",
        "debit",
        "kredit"
    ]

    lower_columns = {
        col.lower(): col
        for col in df.columns
    }

    for col in priority_columns:
        if col in lower_columns:
            return lower_columns[col]

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if len(numeric_columns) > 0:
        return numeric_columns[0]

    return None


def detect_date_column(df):
    """
    Mendeteksi kolom tanggal secara otomatis.
    """

    priority_columns = [
        "tanggal",
        "date",
        "tgl",
        "transaction_date"
    ]

    lower_columns = {
        col.lower(): col
        for col in df.columns
    }

    for col in priority_columns:
        if col in lower_columns:
            return lower_columns[col]

    return None


def dashboard_statistics(result_df):
    """
    Menampilkan statistik utama hasil analisis.
    """

    amount_column = detect_amount_column(result_df)

    total_transaksi = len(result_df)

    total_nominal = 0

    if amount_column is not None:
        total_nominal = pd.to_numeric(
            result_df[amount_column],
            errors="coerce"
        ).fillna(0).sum()

    kategori_terbanyak = (
        result_df["Kategori"]
        .value_counts()
        .idxmax()
    )

    jenis_terbanyak = (
        result_df["Jenis"]
        .value_counts()
        .idxmax()
    )

    pembayaran_terbanyak = (
        result_df["Pembayaran"]
        .value_counts()
        .idxmax()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Jumlah Transaksi",
            total_transaksi
        )

    with col2:
        st.metric(
            "Total Nominal",
            f"Rp {total_nominal:,.0f}"
        )

    with col3:
        st.metric(
            "Kategori Terbanyak",
            kategori_terbanyak
        )

    with col4:
        st.metric(
            "Jenis Terbanyak",
            jenis_terbanyak
        )

    st.info(
        f"Metode pembayaran terbanyak : **{pembayaran_terbanyak}**"
    )


def dashboard_charts(result_df):
    """
    Menampilkan grafik hasil klasifikasi.
    """

    st.subheader("Distribusi Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            result_df["Kategori"]
            .value_counts()
            .reset_index(),
            x="Kategori",
            y="count",
            labels={"count": "Jumlah"},
            title="Distribusi Kategori"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.bar(
            result_df["Pembayaran"]
            .value_counts()
            .reset_index(),
            x="Pembayaran",
            y="count",
            labels={"count": "Jumlah"},
            title="Metode Pembayaran"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            result_df["Jenis"]
            .value_counts()
            .reset_index(),
            x="Jenis",
            y="count",
            labels={"count": "Jumlah"},
            title="Distribusi Jenis"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


def dashboard_table(result_df):
    """
    Menampilkan tabel hasil prediksi.
    """

    st.subheader("Hasil Prediksi")

    st.dataframe(
        result_df,
        use_container_width=True
    )

    csv = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Hasil Prediksi (CSV)",
        data=csv,
        file_name="hasil_prediksi.csv",
        mime="text/csv"
    )


def show_dashboard(result_df):
    """
    Dashboard utama.
    """

    st.header("📊 Dashboard Analisis")

    dashboard_statistics(result_df)

    st.divider()

    dashboard_charts(result_df)

    st.divider()

    dashboard_table(result_df)

# ==========================================================
# BUILD KNOWLEDGE
# ==========================================================
def build_knowledge(result_df):
    """
    Membangun Knowledge Base dari hasil prediksi Machine Learning.
    """

    metadata = []

    for idx, row in result_df.iterrows():

        document = []

        for column in result_df.columns:

            value = row[column]

            if pd.isna(value):
                value = ""

            document.append(
                f"{column}: {value}"
            )

        metadata.append({
            "id": idx,
            "knowledge": "\n".join(document)
        })

    return metadata

# ==========================================================
# BUILD FAISS
# ==========================================================
def build_faiss(
    embeddings
):
    """
    Membangun index FAISS menggunakan embedding IndoBERT.
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


def get_faiss_info(index):
    """
    Mengambil informasi index FAISS.
    """

    return {
        "dimension": index.d,
        "total_vector": index.ntotal
    }


def show_faiss_summary(index):
    """
    Menampilkan ringkasan FAISS.
    """

    info = get_faiss_info(index)

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Jumlah Knowledge",
            info["total_vector"]
        )

    with col2:

        st.metric(
            "Dimensi Embedding",
            info["dimension"]
        )

# ==========================================================
# RETRIEVAL
# ==========================================================
def retrieve_context(
    question,
    tokenizer,
    model,
    faiss_index,
    metadata,
    top_k=5
):
    """
    Melakukan Retrieval menggunakan FAISS
    dan mengembalikan Top-K Knowledge.
    """

    if faiss_index.ntotal == 0:
        return []

    query_embedding = batch_embedding(
        texts=[question],
        tokenizer=tokenizer,
        model=model,
        batch_size=1
    )

    distances, indices = faiss_index.search(
        query_embedding.astype(np.float32),
        top_k
    )

    contexts = []

    for idx in indices[0]:

        if idx == -1:
            continue

        if idx >= len(metadata):
            continue

        contexts.append(
            metadata[idx]["knowledge"]
        )

    return contexts

# ==========================================================
# PROMPT BUILDER
# ==========================================================
def build_prompt(
    question,
    contexts
):
    """
    Membangun prompt untuk Gemini berdasarkan
    hasil Retrieval dari FAISS.
    """

    if len(contexts) == 0:

        context_text = (
            "Tidak ditemukan informasi yang relevan."
        )

    else:

        context_text = "\n\n".join(contexts)

    prompt = f"""
Anda adalah asisten analisis transaksi keuangan.

Jawablah pertanyaan pengguna HANYA berdasarkan
informasi hasil Machine Learning berikut.

Jika informasi tidak tersedia pada context,
jawablah bahwa informasi tersebut tidak ditemukan.

==========================
CONTEXT
==========================

{context_text}

==========================
PERTANYAAN
==========================

{question}

==========================
ATURAN
==========================

1. Jangan membuat informasi baru.
2. Jangan menggunakan pengetahuan di luar context.
3. Jawaban harus jelas, singkat, dan informatif.
4. Gunakan bahasa Indonesia yang baik.
"""

    return prompt


# ==========================================================
# CHATBOT FUNCTION
# ==========================================================
def chatbot_response(
    question,
    tokenizer,
    model,
    faiss_index,
    metadata
):
    """
    Pipeline chatbot RAG.
    """

    contexts = retrieve_context(
        question=question,
        tokenizer=tokenizer,
        model=model,
        faiss_index=faiss_index,
        metadata=metadata,
        top_k=5
    )

    prompt = build_prompt(
        question=question,
        contexts=contexts
    )

    response = gemini_model.generate_content(
        prompt
    )

    return response.text


def initialize_chat():
    """
    Inisialisasi riwayat chat.
    """

    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []


def add_chat(
    role,
    message
):
    """
    Menambahkan pesan ke riwayat chat.
    """

    st.session_state.chat_history.append(
        {
            "role": role,
            "message": message
        }
    )


def show_chat():
    """
    Menampilkan riwayat chat.
    """

    for chat in st.session_state.chat_history:

        with st.chat_message(chat["role"]):

            st.markdown(
                chat["message"]
            )


def chatbot_interface(
    tokenizer,
    model,
    faiss_index,
    metadata
):
    """
    Interface chatbot pada Streamlit.
    """

    st.header(
        "Chatbot Analisis Transaksi"
    )

    initialize_chat()

    show_chat()

    question = st.chat_input(
        "Masukkan pertanyaan..."
    )

    if question:

        add_chat(
            "user",
            question
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Sedang menganalisis..."
            ):

                answer = chatbot_response(
                    question=question,
                    tokenizer=tokenizer,
                    model=model,
                    faiss_index=faiss_index,
                    metadata=metadata
                )

                st.markdown(answer)

        add_chat(
            "assistant",
            answer
        )

# ==========================================================
# MAIN STREAMLIT APP
# ==========================================================
def main():

    st.sidebar.title("📂 Menu")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is None:

        st.info(
            "Silakan upload dataset transaksi terlebih dahulu."
        )

        return

    # ======================================================
    # READ DATASET
    # ======================================================
    try:

        df = read_uploaded_file(
            uploaded_file
        )

        text_column = validate_dataset(
            df
        )

        st.session_state["df"] = df
        st.session_state["text_column"] = text_column

    except Exception as e:

        st.error(
            f"Gagal membaca dataset : {e}"
        )

        return

    # ======================================================
    # DATASET PREVIEW
    # ======================================================
    with st.expander(
        "📄 Preview Dataset",
        expanded=False
    ):

        st.write(
            f"Jumlah Data : **{len(df):,}**"
        )

        st.write(
            f"Kolom Teks : **{text_column}**"
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )

    # ======================================================
    # PROCESS BUTTON
    # ======================================================
    if st.button(
        "🚀 Mulai Analisis",
        use_container_width=True
    ):

        with st.spinner(
            "Melakukan Feature Extraction..."
        ):

            result_df, embeddings = predict_dataframe(
                df=df,
                text_column=text_column,
                tokenizer=tokenizer,
                model=indobert_model
            )

        st.session_state["result_df"] = result_df
        st.session_state["embeddings"] = embeddings

        metadata = build_knowledge(
            result_df
        )

        faiss_index = build_faiss(
            embeddings
        )

        st.session_state["metadata"] = metadata
        st.session_state["faiss_index"] = faiss_index

        st.success(
            "Analisis transaksi berhasil dilakukan."
        )

    # ======================================================
    # DASHBOARD
    # ======================================================
    if "result_df" in st.session_state:

        show_dashboard(
            st.session_state["result_df"]
        )

        st.divider()

        st.subheader(
            "Knowledge Base"
        )

        show_faiss_summary(
            st.session_state["faiss_index"]
        )

        st.divider()

    # ======================================================
    # CHATBOT
    # ======================================================
    if (
        "faiss_index" in st.session_state
        and "metadata" in st.session_state
    ):

        chatbot_interface(
            tokenizer=tokenizer,
            model=indobert_model,
            faiss_index=st.session_state["faiss_index"],
            metadata=st.session_state["metadata"]
        )


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    main()

