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
    """Mengubah label seperti '10000', '10RIBU', atau 'Rp 10.000' menjadi integer nominal."""
    text = str(label).upper().replace(" ", "")
    angka = re.findall(r"\d+", text)

    if not angka:
        return None

    nominal = int("".join(angka))

    # Untuk label seperti 1RIBU, 2RIBU, 10RIBU, dst.
    if "RIBU" in text and nominal < 1000:
        nominal *= 1000

    return nominal


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
        75000: "tujuh puluh lima ribu rupiah",
        100000: "seratus ribu rupiah",
    }

    return mapping.get(nominal, f"{format_rupiah(label)}")


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

    # Ambil ukuran input dari model agar otomatis sesuai hasil training.
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
    top3 = pd.DataFrame(
        {
            "Nominal": [format_rupiah(class_names[i]) for i in top_indices],
            "Confidence": [f"{prediction[i] * 100:.2f}%" for i in top_indices],
        }
    )

    return label_prediksi, label_display, confidence, top3


def render_speech_button(text):
    safe_text = json.dumps(text)
    components.html(
        f"""
        <button onclick="speakResult()" style="
            background-color:#0E1117;
            color:white;
            border:none;
            border-radius:8px;
            padding:10px 16px;
            font-size:15px;
            cursor:pointer;">
            🔊 Putar Suara Hasil Prediksi
        </button>

        <script>
        function speakResult() {{
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance({safe_text});
            msg.lang = "id-ID";
            msg.rate = 0.9;
            msg.pitch = 1;
            window.speechSynthesis.speak(msg);
        }}
        </script>
        """,
        height=70,
    )


# ============================================================
# Tampilan Aplikasi Streamlit
# ============================================================
st.set_page_config(
    page_title="Pengenalan Nominal Uang Rupiah CNN",
    page_icon="💵",
    layout="wide",
)

st.title("Rancang Bangun Aplikasi Pengenalan Nominal Uang Kertas Rupiah Berbasis CNN")
st.write(
    "Aplikasi ini menggunakan model Convolutional Neural Network untuk mengenali "
    "nominal uang kertas Rupiah dari gambar yang diunggah."
)

model, class_names, IMG_SIZE = load_model_and_labels()

with st.sidebar:
    st.header("Informasi Model")
    st.write("Ukuran Input:", f"{IMG_SIZE} x {IMG_SIZE}")
    st.write("Jumlah Kelas:", len(class_names))
    st.write("Daftar Kelas:")
    for class_name in class_names:
        st.write("-", format_rupiah(class_name))

uploaded_files = st.file_uploader(
    "Upload satu atau beberapa gambar uang Rupiah",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True,
)

if uploaded_files:
    hasil_prediksi = []
    kalimat_suara = "Hasil prediksi nominal uang adalah. "

    for nomor, uploaded_file in enumerate(uploaded_files, start=1):
        original_img, resized_img, img_final = preprocess_image(uploaded_file, IMG_SIZE)
        label_prediksi, label_display, confidence, top3 = predict_image(
            model, class_names, img_final
        )

        hasil_prediksi.append(
            {
                "Nama File": uploaded_file.name,
                "Prediksi Nominal": label_display,
                "Confidence": f"{confidence:.2f}%",
            }
        )

        kalimat_suara += (
            f"Gambar ke {nomor} diprediksi sebagai uang "
            f"{nominal_ke_teks(label_prediksi)}, "
            f"dengan tingkat kepercayaan {confidence:.2f} persen. "
        )

        st.divider()
        st.subheader(f"Hasil Prediksi Gambar {nomor}")

        col1, col2, col3 = st.columns([1.2, 1.2, 1])

        with col1:
            st.image(original_img, caption="Gambar Asli", use_container_width=True)

        with col2:
            st.image(resized_img, caption=f"Gambar Setelah Resize {IMG_SIZE} x {IMG_SIZE}", use_container_width=True)

        with col3:
            st.metric("Prediksi Nominal", label_display)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.write("Top 3 Prediksi")
            st.dataframe(top3, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Ringkasan Hasil Prediksi")

    df_hasil = pd.DataFrame(hasil_prediksi)
    st.dataframe(df_hasil, use_container_width=True, hide_index=True)

    csv_buffer = StringIO()
    df_hasil.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Hasil Prediksi CSV",
        data=csv_buffer.getvalue(),
        file_name="hasil_prediksi_uang_rupiah.csv",
        mime="text/csv",
    )

    st.subheader("Output Suara")
    render_speech_button(kalimat_suara)
else:
    st.info("Silakan upload gambar uang Rupiah untuk melakukan prediksi.")
