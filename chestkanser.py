import streamlit as st
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
# Load the pre-trained model
model = tf.keras.models.load_model("C:/Users/can/Desktop/chest/chest.h5")

def predict_class(image_path):
    # Load the image and resize it to 224x224 pixels
    img = image.load_img(image_path, target_size=(224, 224))
    # Convert the image to a numpy array and normalize it
    imag = image.img_to_array(img)
    # Reshape the array to (1, 224, 224, 3) to feed into the model
    imaga = np.expand_dims(imag, axis=0)
    # Use the model to make predictions
    y_pred = model.predict(imaga)
    # Get the predicted class label
    class_idx = np.argmax(y_pred,-1)
    if class_idx == 0:
        class_label = "Adenocarcinoma"
    elif class_idx == 1:
        class_label = "Large cell carcinoma"
    elif class_idx == 2:
        class_label = "Normal (void of cancer)"
    else:
        class_label = "Squamous cell carcinoma"
    return class_label

# Set app title and favicon
st.set_page_config(page_title="Chest Cancer Classifier", page_icon=":microscope:")

# Render the web app
def main():
    st.title("Chest Cancer Classifier")
    st.write("This app predicts the type of lung cancer (if any) in a chest X-ray image.")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_path = "uploaded_image.png"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        class_label = predict_class(image_path)
        st.write("Prediction: ", class_label)
        st.image(image_path)

if __name__ == "__main__":
    main()
