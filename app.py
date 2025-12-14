import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.title('Classificador KMNIST')


model = tf.keras.models.load_model('models/cnn_seed42.h5')


uploaded = st.file_uploader('Envie uma imagem 28x28 em escala de cinza')


if uploaded:
    img = Image.open(uploaded).convert('L').resize((28,28))
    img_arr = np.array(img) / 255.0
    img_arr = img_arr.reshape(1,28,28,1)


pred = model.predict(img_arr)
st.write('Classe predita:', np.argmax(pred))