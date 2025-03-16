import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os


MODEL_PATH = 'potato_leaf_model.h5'  
model = load_model(r"C:\Users\Dell\OneDrive\Desktop\Potato leaf dataset\Potato_leaf_dieases.keras.h5")

CLASS_NAMES = ['Healthy', 'Early Blight', 'Late Blight']


def preprocess_image(img):
    img = img.resize((150,150))  
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

st.title(' Potato Leaf Disease Prediction')


uploaded_file = st.file_uploader("Upload an image of a potato leaf...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    
    img = Image.open(uploaded_file)
    processed_img = preprocess_image(img)

    
    prediction = model.predict(processed_img)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    
    st.write(f"###  Predicted Disease: **{predicted_class}**")
    st.write(f"###  Confidence: **{confidence:.2%}**")

    
    st.write("###  Recommendations:")
    if predicted_class == 'Healthy':
        st.success(" The plant is healthy. Keep monitoring regularly.")
    elif predicted_class == 'Early Blight':
        st.warning(" Early Blight detected. Suggested treatment: Apply fungicides like chlorothalonil.")
    elif predicted_class == 'Late Blight':
        st.error(" Late Blight detected. Suggested treatment: Remove affected leaves and apply systemic fungicides.")

#

