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
    """
    Mengubah label seperti:
    '10000', '100000', '10RIBU', '100K', 'Rp 10.000'
    menjadi integer nominal.
    """
    text = str(label).upper().replace(" ", "")

    valid_nominal = [
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        75000,
        100000
    ]

    text_clean = re.sub(r"[^0-9A-Z]", "", text)

    # Cek nominal penuh, contoh: 100000, 50000, 20000
    for nominal in sorted(valid_nominal, reverse=True):
        if str(nominal) in text_clean:
            return nominal

    # Cek format 1RIBU, 10RIBU, 100K, 50K, 20RB
    match = re.search(r"(\d+)(RIBU|RB|K)", text_clean)

    if match:
        angka = int(match.group(1))
        nominal = angka * 1000

        if nominal in valid_nominal:
            return nominal

    # Cek angka biasa
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
        75000: "tujuh puluh lima ribu rupiah",
        100000: "seratus ribu rupiah"
    }

    return mapping.get(nominal, str(label))


def angka_ke_teks(angka):
    angka = int(round(float(angka)))

    satuan = [
        "nol",
        "satu",
        "dua",
        "tiga",
        "empat",
        "lima",
        "enam",
        "tujuh",
        "delapan",
        "sembilan",
        "sepuluh",
        "sebelas"
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
        height=80
    )


# ============================================================
# Tampilan Aplikasi Streamlit
# ============================================================

st.set_page_config(
    page_title="Pengenalan Nominal Uang Rupiah CNN",
    page_icon="💵",
    layout="wide"
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
    accept_multiple_files=True
)


if uploaded_files:
    hasil_prediksi = []
    kalimat_suara = "Hasil prediksi nominal uang adalah. "

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

        st.divider()
        st.subheader(f"Hasil Prediksi Gambar {nomor}")

        col1, col2, col3 = st.columns([1.2, 1.2, 1])

        with col1:
            st.image(
                original_img,
                caption="Gambar Asli",
                use_container_width=True
            )

        with col2:
            st.image(
                resized_img,
                caption=f"Gambar Setelah Resize {IMG_SIZE} x {IMG_SIZE}",
                use_container_width=True
            )

        with col3:
            st.metric("Prediksi Nominal", label_display)
            st.metric("Confidence", f"{confidence:.2f}%")

            st.write("Top 3 Prediksi")
            st.dataframe(
                top3,
                use_container_width=True,
                hide_index=True
            )

    st.divider()
    st.subheader("Ringkasan Hasil Prediksi")

    df_hasil = pd.DataFrame(hasil_prediksi)

    st.dataframe(
        df_hasil,
        use_container_width=True,
        hide_index=True
    )

    csv_buffer = StringIO()
    df_hasil.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Hasil Prediksi CSV",
        data=csv_buffer.getvalue(),
        file_name="hasil_prediksi_uang_rupiah.csv",
        mime="text/csv"
    )

    st.subheader("Output Suara")

    st.info(kalimat_suara)

    render_speech_button(kalimat_suara)

else:
    st.info("Silakan upload gambar uang Rupiah untuk melakukan prediksi.")
