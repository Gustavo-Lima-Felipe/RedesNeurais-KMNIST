import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# Configuração da página
# =========================
st.set_page_config(
    page_title="KMNIST Classifier",
    layout="centered"
)

st.title("🈴 Classificador KMNIST")
st.write("Envie uma imagem e o modelo vai identificar o caractere japonês.")


# =========================
# Carregar modelo
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/cnn_seed42.h5")

model = load_model()


# =========================
# Classes do KMNIST
# =========================
CLASS_NAMES = [
    "o", "ki", "su", "tsu", "na",
    "ha", "ma", "ya", "re", "wo"
]


# =========================
# Upload da imagem
# =========================
uploaded_file = st.file_uploader(
    "Envie uma imagem (PNG, JPG ou JPEG)",
    type=["png", "jpg", "jpeg"]
)


# =========================
# Processamento e predição
# =========================
if uploaded_file is not None:

    # Abrir imagem
    image = Image.open(uploaded_file).convert("L")

    st.subheader("Imagem enviada")
    st.image(image, width=200)

    # Pré-processamento
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Predição
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Resultado
    st.subheader("Resultado")
    st.write(f"**Classe prevista:** {CLASS_NAMES[predicted_class]}")
    st.write(f"**Confiança:** {confidence * 100:.2f}%")

    # Gráfico de probabilidades
    st.subheader("Probabilidades por classe")
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, predictions[0])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilidade")
    st.pyplot(fig)

    # Botão de reset
    if st.button("Limpar imagem"):
        st.experimental_rerun()
