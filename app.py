import json
import os
import re
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image, ImageOps

# ============================================================
# Konfigurasi File Model Dan Label
# ============================================================

MODEL_PATH = "model_pengenalan_uang_rupiah_cnn.h5"
LABEL_PATH = "class_names_uang_rupiah.json"


# ============================================================
# Fungsi Bantuan
# ============================================================

def label_to_nominal_int(label):
    text = str(label).upper().replace(" ", "")
    valid_nominal = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    text_clean = re.sub(r"[^0-9A-Z]", "", text)

    for nominal in sorted(valid_nominal, reverse=True):
        if str(nominal) in text_clean:
            return nominal

    match = re.search(r"(\d+)(RIBU|RB|K)", text_clean)
    if match:
        angka = int(match.group(1))
        nominal = angka * 1000
        if nominal in valid_nominal:
            return nominal

    angka_list = re.findall(r"\d+", text_clean)
    for angka in angka_list:
        nilai = int(angka)
        if nilai in valid_nominal:
            return nilai
        if nilai * 1000 in valid_nominal:
            return nilai * 1000

    return None


def format_rupiah(label):
    nominal = label_to_nominal_int(label)
    if nominal is None:
        return str(label)
    return "Rp " + f"{nominal:,}".replace(",", ".")


def nominal_ke_teks(label):
    nominal = label_to_nominal_int(label)
    mapping = {
        1000: "seribu rupiah",
        2000: "dua ribu rupiah",
        5000: "lima ribu rupiah",
        10000: "sepuluh ribu rupiah",
        20000: "dua puluh ribu rupiah",
        50000: "lima puluh ribu rupiah",
        100000: "seratus ribu rupiah"
    }
    return mapping.get(nominal, str(label))


def angka_ke_teks(angka):
    angka = int(round(float(angka)))
    satuan = [
        "nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "sebelas"
    ]
    if angka < 12:
        return satuan[angka]
    if angka < 20:
        return satuan[angka - 10] + " belas"
    if angka < 100:
        puluh = angka // 10
        sisa = angka % 10
        if sisa == 0:
            return satuan[puluh] + " puluh"
        return satuan[puluh] + " puluh " + satuan[sisa]
    if angka == 100:
        return "seratus"
    return str(angka)


def confidence_ke_teks(confidence):
    confidence_bulat = round(float(confidence))
    return angka_ke_teks(confidence_bulat) + " persen"


@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan: {MODEL_PATH}")
        st.stop()

    if not os.path.exists(LABEL_PATH):
        st.error(f"File label tidak ditemukan: {LABEL_PATH}")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    input_shape = model.input_shape
    img_size = input_shape[1] if input_shape[1] is not None else 160

    return model, class_names, img_size


def preprocess_image(uploaded_file, img_size):
    original_img = Image.open(uploaded_file)
    original_img = ImageOps.exif_transpose(original_img).convert("RGB")

    resized_img = original_img.resize((img_size, img_size))
    img_array = np.array(resized_img).astype("float32") / 255.0
    img_final = np.expand_dims(img_array, axis=0)

    return original_img, resized_img, img_final


def predict_image(model, class_names, img_final):
    prediction = model.predict(img_final, verbose=0)[0]

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    label_prediksi = class_names[predicted_class]
    label_display = format_rupiah(label_prediksi)

    top_indices = np.argsort(prediction)[::-1][:3]

    top3 = pd.DataFrame({
        "Nominal": [format_rupiah(class_names[i]) for i in top_indices],
        "Confidence": [f"{prediction[i] * 100:.2f}%" for i in top_indices]
    })

    return label_prediksi, label_display, confidence, top3


def render_speech_button(text):
    safe_text = json.dumps(text)

    components.html(
        f"""
        <button onclick="speakResult()" style="
            background: linear-gradient(135deg, #0f766e 0%, #0284c7 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 12px 18px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 10px 24px rgba(2, 132, 199, 0.22);
        ">
            🔊 Putar Suara Hasil Prediksi
        </button>

        <script>
        function getIndonesianVoice() {{
            const voices = window.speechSynthesis.getVoices();

            let voice = voices.find(v =>
                v.lang.toLowerCase().startsWith("id")
            );

            if (!voice) {{
                voice = voices.find(v =>
                    v.name.toLowerCase().includes("indonesia")
                );
            }}

            if (!voice) {{
                voice = voices.find(v =>
                    v.lang.toLowerCase().startsWith("ms")
                );
            }}

            return voice || null;
        }}

        function speakResult() {{
            window.speechSynthesis.cancel();

            const msg = new SpeechSynthesisUtterance({safe_text});
            msg.lang = "id-ID";
            msg.rate = 0.85;
            msg.pitch = 1;

            const selectedVoice = getIndonesianVoice();

            if (selectedVoice) {{
                msg.voice = selectedVoice;
            }}

            window.speechSynthesis.speak(msg);
        }}

        window.speechSynthesis.onvoiceschanged = function() {{
            getIndonesianVoice();
        }};
        </script>
        """,
        height=85
    )


# ============================================================
# Tampilan Aplikasi Streamlit
# ============================================================

st.set_page_config(
    page_title="Pengenalan Nominal Uang Rupiah CNN",
    page_icon="💵",
    layout="wide"
)

# ============================================================
# Custom CSS untuk Responsivitas
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f6f8fb 0%, #eef4ff 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        font-size: 15px;
    }

    /* Hero Card */
    .hero-card {
        background: linear-gradient(135deg, #0f766e 0%, #0ea5e9 100%);
        padding: 36px 40px;
        border-radius: 26px;
        color: white;
        box-shadow: 0 18px 45px rgba(15, 118, 110, 0.22);
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 14px;
    }

    .hero-subtitle {
        font-size: 17px;
        line-height: 1.7;
        opacity: 0.95;
        max-width: 850px;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 22px;
    }

    .badge {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.28);
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
    }

    /* Custom Card */
    .custom-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(226,232,240,0.9);
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 10px 32px rgba(15, 23, 42, 0.08);
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 8px;
    }

    .section-desc {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 18px;
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #38bdf8;
        border-radius: 20px;
        padding: 18px;
    }

    div[data-testid="stFileUploader"] label {
        font-weight: 700;
        color: #0f172a;
    }

    /* Metric */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
        color: #0f766e;
        font-weight: 900;
    }

    .stDownloadButton button {
        background: linear-gradient(135deg, #0f766e 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 12px 20px;
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(2, 132, 199, 0.22);
    }

    .stDownloadButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(2, 132, 199, 0.28);
    }

    div[data-testid="stAlert"] {
        border-radius: 18px;
        border: none;
    }

    img {
        border-radius: 18px;
        max-width: 100%; /* Pastikan gambar responsif */
    }

    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* Responsif untuk layar kecil */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 30px;
        }

        .hero-card {
            padding: 28px 24px;
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        .custom-card {
            padding: 18px;
        }

        .badge {
            font-size: 12px;
            padding: 6px 12px;
        }

        .badge-row {
            gap: 8px;
        }

        div[data-testid="stFileUploader"] {
            padding: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Load Model
# ============================================================

model, class_names, IMG_SIZE = load_model_and_labels()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 💵 Informasi Model")
    st.markdown("---")

    st.markdown(
        f"""
        <div class="info-model-box">
            <div class="info-model-item">
                <div class="info-model-label">Ukuran Input</div>
                <div class="info-model-value">{IMG_SIZE} x {IMG_SIZE}</div>
            </div>

            <div class="info-model-item">
                <div class="info-model-label">Jumlah Kelas</div>
                <div class="info-model-value">{len(class_names)} kelas</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Daftar Nominal")

    for class_name in class_names:
        st.markdown(f"- {format_rupiah(class_name)}")

    st.markdown("---")
    st.caption("Aplikasi CNN untuk klasifikasi nominal uang kertas Rupiah.")


# ============================================================
# Hero Section
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">
            Pengenalan Nominal Uang Kertas Rupiah Berbasis CNN
        </div>
        <div class="hero-subtitle">
            Upload gambar uang kertas Rupiah, lalu sistem akan mengenali nominalnya
            menggunakan model Convolutional Neural Network. Hasil prediksi ditampilkan
            lengkap dengan tingkat kepercayaan dan tiga kemungkinan tertinggi.
        </div>
        <div class="badge-row">
            <div class="badge">📷 Input Gambar</div>
            <div class="badge">🧠 CNN Model</div>
            <div class="badge">📊 Top 3 Prediksi</div>
            <div class="badge">🔊 Output Suara</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Upload Section
# ============================================================

st.markdown(
    """
    <div class="custom-card">
        <div class="section-title">Upload Gambar Uang Rupiah</div>
        <div class="section-desc">
            Pilih satu atau beberapa gambar uang kertas Rupiah dengan format JPG, PNG, WEBP, atau BMP.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Masukkan gambar untuk diprediksi",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)


# ============================================================
# Proses Prediksi
# ============================================================

if uploaded_files:
    hasil_prediksi = []
    kalimat_suara = "Hasil prediksi nominal uang adalah. "

    st.markdown("## 📌 Hasil Prediksi")

    for nomor, uploaded_file in enumerate(uploaded_files, start=1):
        original_img, resized_img, img_final = preprocess_image(uploaded_file, IMG_SIZE)

        label_prediksi, label_display, confidence, top3 = predict_image(
            model,
            class_names,
            img_final
        )

        hasil_prediksi.append({
            "Nama File": uploaded_file.name,
            "Prediksi Nominal": label_display,
            "Confidence": f"{confidence:.2f}%"
        })

        kalimat_suara += (
            f"Gambar ke {angka_ke_teks(nomor)}. "
            f"Prediksi nominal uang adalah {nominal_ke_teks(label_prediksi)}. "
            f"Tingkat kepercayaan {confidence_ke_teks(confidence)}. "
        )

        st.markdown(
            f"""
            <div class="custom-card">
                <div class="section-title">Gambar {nomor}: {uploaded_file.name}</div>
                <div class="section-desc">
                    Berikut hasil klasifikasi nominal uang berdasarkan gambar yang diunggah.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1.2, 1])

        with col1:
            img_col1, img_col2 = st.columns(2)

            with img_col1:
                st.image(
                    original_img,
                    caption="Gambar Asli",
                    use_container_width=True
                )

            with img_col2:
                st.image(
                    resized_img,
                    caption=f"Resize {IMG_SIZE} x {IMG_SIZE}",
                    use_container_width=True
                )

        with col2:
            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric("Prediksi", label_display)

            with metric_col2:
                st.metric("Confidence", f"{confidence:.2f}%")

            st.markdown("#### Top 3 Prediksi")
            st.dataframe(
                top3,
                use_container_width=True,
                hide_index=True
            )

        st.divider()


    # ========================================================
    # Ringkasan Hasil
    # ========================================================

    st.markdown("## 📋 Ringkasan Hasil Prediksi")

    df_hasil = pd.DataFrame(hasil_prediksi)

    st.dataframe(
        df_hasil,
        use_container_width=True,
        hide_index=True
    )

    csv_buffer = StringIO()
    df_hasil.to_csv(csv_buffer, index=False)

    st.download_button(
        label="⬇️ Download Hasil Prediksi CSV",
        data=csv_buffer.getvalue(),
        file_name="hasil_prediksi_uang_rupiah.csv",
        mime="text
