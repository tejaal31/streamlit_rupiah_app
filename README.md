# Aplikasi Pengenalan Nominal Uang Kertas Rupiah Berbasis CNN

Aplikasi Streamlit untuk melakukan prediksi nominal uang kertas Rupiah menggunakan model CNN hasil training.

## Struktur File

```text
repo-name/
├── app.py
├── requirements.txt
├── model_pengenalan_uang_rupiah_cnn.h5
└── class_names_uang_rupiah.json
```

## Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cara Deploy Ke Streamlit Community Cloud

1. Buat repository GitHub baru.
2. Upload `app.py`, `requirements.txt`, `model_pengenalan_uang_rupiah_cnn.h5`, dan `class_names_uang_rupiah.json`.
3. Buka Streamlit Community Cloud.
4. Pilih repository GitHub.
5. Isi main file path dengan `app.py`.
6. Klik Deploy.

## Catatan

Pastikan nama file model dan label sama dengan yang digunakan di `app.py`:

```text
model_pengenalan_uang_rupiah_cnn.h5
class_names_uang_rupiah.json
```
