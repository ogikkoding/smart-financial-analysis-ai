# ==========================================================
# IMPORT LIBRARIES
# ==========================================================
import gc
import os
import re
import faiss
import google.generativeai as genai
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import xgboost as xgb
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

# ==========================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="Sistem Analisis Transaksi Keuangan Cerdas",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# CSS CUSTOM MODERN (DARK MODE EMERALD & GLASSMORPHISM)
# ==========================================================
st.markdown(
    """
<style>
.stApp { background: linear-gradient(135deg, #0F172A, #111827, #000000); }
.header-container {
    background: linear-gradient(135deg, rgba(0, 137, 123, 0.18), rgba(38, 166, 154, 0.05));
    padding: 30px; border-radius: 20px; border: 1px solid rgba(38, 166, 154, 0.3);
    box-shadow: 0px 8px 25px rgba(0, 137, 123, 0.15); backdrop-filter: blur(10px); margin-bottom: 25px;
}
.main-title { text-align: center; color: #26A69A; font-size: 38px; font-weight: 700; margin-bottom: 8px; }
.sub-title { text-align: center; color: #9CA3AF; font-size: 17px; margin-bottom: 0px; }
[data-testid="stMetricValue"] { font-size: 24px !important; color: #26A69A !important; font-weight: bold; }
[data-testid="stFileUploader"] { border: 2px dashed #00897B; border-radius: 15px; padding: 20px; background: rgba(15, 23, 42, 0.6); }
.stButton>button {
    width: 100%; background: linear-gradient(90deg, #26A69A, #00897B); color: white;
    border: none; border-radius: 12px; font-size: 17px; font-weight: bold; padding: 12px; transition: 0.3s;
}
.stButton>button:hover { background: linear-gradient(90deg, #00897B, #00695C); transform: scale(1.01); }
[data-testid="stChatMessage"] { background-color: rgba(30, 41, 59, 0.7); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.05); padding: 12px; margin-bottom: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# BASE PATH DEFINITION
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "MODELS")
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.join(BASE_DIR, "..", "MODELS")

# ==========================================================
# KONFIGURASI GEMINI API (FLEKSIBEL AUTO-FALLBACK)
# ==========================================================
load_dotenv()
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def get_gemini_response(prompt):
    """Mencoba memanggil model Gemini secara berurutan sampai menemukan yang aktif/free."""
    available_models = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-pro",
    ]

    for model_name in available_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            continue

    return "⚠️ Maaf Gik, semua model Gemini saat ini sedang tidak dapat diakses. Coba periksa kembali API Key kamu."


def chatbot_response(question, tokenizer, model, faiss_index, metadata):
    # Cek kunci API langsung menggunakan GEMINI_API_KEY
    if not GEMINI_API_KEY:
        return "⚠️ API Key Gemini belum terkonfigurasi di Streamlit Secrets."

    contexts = retrieve_context(
        question, tokenizer, model, faiss_index, metadata
    )
    context_text = (
        "\n\n".join(contexts)
        if contexts
        else "Tidak ditemukan informasi relevan."
    )

    prompt = f"""
Anda adalah Asisten AI Analisis Transaksi Keuangan Cerdas.
Jawablah pertanyaan pengguna HANYA berdasarkan CONTEXT data transaksi berikut.

CONTEXT:
{context_text}

PERTANYAAN:
{question}

ATURAN:
1. Jawab singkat, akurat, dan ramah.
2. Jangan membuat asumsi atau informasi baru di luar CONTEXT.
3. Gunakan Bahasa Indonesia yang jelas.
"""
    return get_gemini_response(prompt)

# ==========================================================
# LOAD MODEL INDOBERT, XGBOOST, ENCODERS & SCALERS (CACHED)
# ==========================================================
INDOBERT_MODEL_NAME = "indobenchmark/indobert-base-p1"

@st.cache_resource
def load_indobert():
    tokenizer = AutoTokenizer.from_pretrained(INDOBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(INDOBERT_MODEL_NAME)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_xgboost_models():
    model_kategori = xgb.XGBClassifier()
    model_kategori.load_model(os.path.join(MODEL_DIR, "xgb_model_kategori.json"))
    model_jenis = xgb.XGBClassifier()
    model_jenis.load_model(os.path.join(MODEL_DIR, "xgb_model_jenis.json"))
    model_pembayaran = xgb.XGBClassifier()
    model_pembayaran.load_model(os.path.join(MODEL_DIR, "xgb_model_pembayaran.json"))
    return {"kategori": model_kategori, "jenis": model_jenis, "pembayaran": model_pembayaran}

@st.cache_resource
def load_encoders():
    return {
        "kategori": np.load(os.path.join(MODEL_DIR, "encoder_kategori.npy"), allow_pickle=True),
        "jenis": np.load(os.path.join(MODEL_DIR, "encoder_jenis.npy"), allow_pickle=True),
        "pembayaran": np.load(os.path.join(MODEL_DIR, "encoder_pembayaran.npy"), allow_pickle=True),
    }

@st.cache_resource
def load_scalers():
    return {
        "kategori": joblib.load(os.path.join(MODEL_DIR, "scaler_kategori.joblib")),
        "jenis": joblib.load(os.path.join(MODEL_DIR, "scaler_jenis.joblib")),
        "pembayaran": joblib.load(os.path.join(MODEL_DIR, "scaler_pembayaran.joblib")),
    }

tokenizer, indobert_model = load_indobert()
xgb_models = load_xgboost_models()
encoders = load_encoders()
scalers = load_scalers()

# ==========================================================
# UTILITY & PREPROCESSING FUNCTIONS
# ==========================================================
def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format file tidak didukung.")

def validate_dataset(df):
    if df is None or df.empty:
        raise ValueError("Dataset kosong atau tidak valid.")
    cols = df.columns.tolist()
    lower_cols = {str(c).lower(): c for c in cols}

    desc_col = next((lower_cols[p] for p in ["deskripsi transaksi", "deskripsi", "keterangan", "uraian", "description"] if p in lower_cols), None)
    merchant_col = next((lower_cols[p] for p in ["merchant", "toko", "vendor", "pembayaran ke"] if p in lower_cols), None)
    amount_col = next((lower_cols[p] for p in ["jumlah_rp", "jumlah", "nominal", "amount", "nilai", "total"] if p in lower_cols), None)

    return {"merchant": merchant_col, "deskripsi": desc_col, "jumlah": amount_col}

def safe_text(text):
    return "" if pd.isna(text) else str(text)

def preprocess_text(text):
    text = safe_text(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    return pooled / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def batch_embedding(texts, tokenizer, model, batch_size=16, max_length=128):
    all_embeddings = []
    texts = [preprocess_text(text) for text in texts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            embedding = mean_pooling(output, encoded["attention_mask"]).cpu().numpy().astype(np.float32)
            all_embeddings.append(embedding)
            
    # Clean memory
    gc.collect()
    return np.vstack(all_embeddings)

# ==========================================================
# INFERENCE PIPELINE
# ==========================================================
def decode_prediction(predictions, encoder):
    predictions = np.asarray(predictions).astype(int)
    return encoder[predictions]

def predict_dataframe(df, detected_cols, tokenizer, model):
    result_df = df.copy()

    merchant_str = result_df[detected_cols["merchant"]].fillna("").astype(str) if detected_cols["merchant"] else ""
    deskripsi_str = result_df[detected_cols["deskripsi"]].fillna("").astype(str) if detected_cols["deskripsi"] else ""
    teks_input = "Merchant: " + merchant_str + ". Deskripsi: " + deskripsi_str

    embeddings_768 = batch_embedding(teks_input.tolist(), tokenizer, model)

    if detected_cols["jumlah"]:
        jumlah_num = pd.to_numeric(result_df[detected_cols["jumlah"]], errors="coerce").fillna(0).values.reshape(-1, 1)
    else:
        jumlah_num = np.zeros((len(result_df), 1))

    num_kat = scalers["kategori"].transform(jumlah_num)
    feat_kat = np.hstack((embeddings_768, num_kat))

    num_jen = scalers["jenis"].transform(jumlah_num)
    feat_jen = np.hstack((embeddings_768, num_jen))

    num_pem = scalers["pembayaran"].transform(jumlah_num)
    feat_pem = np.hstack((embeddings_768, num_pem))

    pred_kat = xgb_models["kategori"].predict(feat_kat)
    pred_jen = xgb_models["jenis"].predict(feat_jen)
    pred_pem = xgb_models["pembayaran"].predict(feat_pem)

    result_df["Kategori_Prediksi"] = decode_prediction(pred_kat, encoders["kategori"])
    result_df["Jenis_Prediksi"] = decode_prediction(pred_jen, encoders["jenis"])
    result_df["Transaksi_Prediksi"] = decode_prediction(pred_pem, encoders["pembayaran"])
    return result_df, embeddings_768

# ==========================================================
# RAG KNOWLEDGE BASE & FAISS
# ==========================================================
def build_knowledge(result_df):
    metadata = []
    for idx, row in result_df.iterrows():
        doc = [f"{col}: {row[col]}" for col in result_df.columns if not pd.isna(row[col])]
        metadata.append({"id": idx, "knowledge": "\n".join(doc)})
    return metadata

def build_faiss(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_context(question, tokenizer, model, faiss_index, metadata, top_k=5):
    if faiss_index.ntotal == 0:
        return []
    query_emb = batch_embedding([question], tokenizer, model, batch_size=1)
    distances, indices = faiss_index.search(query_emb.astype(np.float32), top_k)
    contexts = [metadata[idx]["knowledge"] for idx in indices[0] if idx != -1 and idx < len(metadata)]
    return contexts

def chatbot_response(question, tokenizer, model, faiss_index, metadata):
    if not gemini_model:
        return "⚠️ API Key Gemini belum terkonfigurasi di Streamlit Secrets."

    contexts = retrieve_context(question, tokenizer, model, faiss_index, metadata)
    context_text = "\n\n".join(contexts) if contexts else "Tidak ditemukan informasi relevan."

    prompt = f"""
Anda adalah Asisten AI Analisis Transaksi Keuangan Cerdas.
Jawablah pertanyaan pengguna HANYA berdasarkan CONTEXT data transaksi berikut.

CONTEXT:
{context_text}

PERTANYAAN:
{question}

ATURAN:
1. Jawab singkat, akurat, dan ramah.
2. Jangan membuat asumsi atau informasi baru di luar CONTEXT.
3. Gunakan Bahasa Indonesia yang jelas.
"""
    response = gemini_model.generate_content(prompt)
    return response.text

# ==========================================================
# DASHBOARD COMPONENTS & CHATBOT
# ==========================================================
def show_dashboard(result_df, detected_cols):
    st.header("📊 Dashboard Analisis Transaksi")
    amount_col = detected_cols["jumlah"]
    total_transaksi = len(result_df)
    total_nominal = pd.to_numeric(result_df[amount_col], errors="coerce").fillna(0).sum() if amount_col else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transaksi", f"{total_transaksi:,}")
    c2.metric("Total Nominal", f"Rp {total_nominal:,.0f}")
    c3.metric("Kategori Terbanyak", result_df["Kategori_Prediksi"].value_counts().idxmax())
    c4.metric("Jenis Terbanyak", result_df["Jenis_Prediksi"].value_counts().idxmax())

    st.info(f"💳 Transaksi Dominan: **{result_df['Transaksi_Prediksi'].value_counts().idxmax()}**")
    st.divider()

    st.subheader("Visualisasi Hasil Prediksi")
    g1, g2 = st.columns(2)
    with g1:
        fig1 = px.bar(result_df["Kategori_Prediksi"].value_counts().reset_index(), x="Kategori_Prediksi", y="count", color="Kategori_Prediksi", title="Distribusi Kategori Transaksi", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        fig2 = px.pie(result_df["Jenis_Prediksi"].value_counts().reset_index(), names="Jenis_Prediksi", values="count", title="Proporsi Jenis Transaksi", hole=0.4, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Data Hasil Klasifikasi")
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(label="📥 Download Hasil Prediksi (CSV)", data=csv, file_name="hasil_klasifikasi_transaksi.csv", mime="text/csv")

def chatbot_interface(tokenizer, model, faiss_index, metadata):
    st.header("💬 Chatbot AI Analisis Transaksi (RAG)")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["message"])

    question = st.chat_input("Tanyakan sesuatu tentang transaksi kamu...")
    if question:
        st.session_state.chat_history.append({"role": "user", "message": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Menganalisis Knowledge Base..."):
                answer = chatbot_response(question, tokenizer, model, faiss_index, metadata)
                st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "message": answer})

# ==========================================================
# MAIN APPLICATION ROUTE
# ==========================================================
def main():
    st.markdown("""<div class="header-container"><p class='main-title'>💰 Sistem Analisis Transaksi Keuangan Cerdas</p><p class='sub-title'>Hybrid IndoBERT + XGBoost Classification & Retrieval-Augmented Generation (RAG) Chatbot</p></div>""", unsafe_allow_html=True)

    st.sidebar.title("📂 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset Transaksi", type=["csv", "xlsx", "xls"])

    if uploaded_file is None:
        st.info("👋 **Selamat datang,** Silakan upload dataset transaksi (`.csv` / `.xlsx`) pada sidebar untuk memulai.")
        return

    try:
        df = read_uploaded_file(uploaded_file)
        detected_cols = validate_dataset(df)
        st.session_state["df"] = df
        st.session_state["detected_cols"] = detected_cols
    except Exception as e:
        st.error(f"Gagal membaca dataset: {e}")
        return

    with st.expander("🔍 Preview Dataset yang Di-upload", expanded=False):
        st.write(f"Total Baris Data: **{len(df):,}** | Kolom Terdeteksi: Merchant: `{detected_cols['merchant']}`, Deskripsi: `{detected_cols['deskripsi']}`, Jumlah: `{detected_cols['jumlah']}`")
        st.dataframe(df.head(), use_container_width=True)

    if st.button("🚀 Mulai Analisis ML & RAG", use_container_width=True):
        with st.spinner("Sedang memproses Feature Extraction & Klasifikasi XGBoost..."):
            result_df, embeddings = predict_dataframe(df, detected_cols, tokenizer, indobert_model)

        st.session_state["result_df"] = result_df
        st.session_state["embeddings"] = embeddings
        st.session_state["metadata"] = build_knowledge(result_df)
        st.session_state["faiss_index"] = build_faiss(embeddings)
        st.success("🎉 Analisis transaksi dan pembentukan Knowledge Base RAG berhasil!")

    if "result_df" in st.session_state:
        st.divider()
        show_dashboard(st.session_state["result_df"], st.session_state["detected_cols"])

    if "faiss_index" in st.session_state and "metadata" in st.session_state:
        st.divider()
        chatbot_interface(tokenizer, indobert_model, st.session_state["faiss_index"], st.session_state["metadata"])

    st.divider()
    st.markdown("""<div style="text-align:center; color:#808080; font-size:14px;"><b>Intelligent Financial Transaction Analysis System</b><br>Developed by <b>Yogi Irawan</b></div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()