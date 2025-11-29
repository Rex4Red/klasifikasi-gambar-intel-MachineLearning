import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# === KONFIGURASI ===
MODEL_PATH = 'submission_model.h5'
# Sesuaikan nama kelas dengan urutan folder dataset-mu (Alphabetical)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# === LOAD MODEL ===
# Kita pakai caching supaya model tidak diload berulang-ulang setiap ada user input
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# === UI WEBSITE ===
st.title("🌲 Intel Image Classification 🏙️")
st.write("Upload gambar pemandangan (Gunung, Laut, Hutan, Kota, dll) untuk diklasifikasi.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)
    
    # Tombol Prediksi
    if st.button('Klasifikasi Gambar'):
        with st.spinner('Sedang memproses...'):
            # Preprocessing (Sama persis dengan saat training)
            img = image.resize((150, 150))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = x / 255.0  # Normalisasi
            x = np.expand_dims(x, axis=0)
            
            # Prediksi
            prediction = model.predict(x)
            index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            result = CLASS_NAMES[index]
            
            # Tampilkan Hasil
            st.success(f"Prediksi: **{result.upper()}**")
            st.info(f"Tingkat Keyakinan: {confidence:.2f}%")
            
            # Tampilkan grafik probabilitas
            st.bar_chart(prediction[0])