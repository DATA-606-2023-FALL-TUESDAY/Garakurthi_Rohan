import os, sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from scipy.special import softmax

# Load the model
loaded_model = tf.keras.saving.load_model('nasnet_mobile.h5')

# Define class names
classes = ['Bus', 'Car', 'Truck', 'motorcycle', 'non-vehicle', 'planes', 'trains']

# Define the function to make predictions on an image
def get_predictions(image):
    try: 
        image = image.resize(size=(224, 224))
        image_arr = np.array(image).reshape(1, 224, 224, 3)
        # Prediction
        # Make a prediction on the image
        img_probas = softmax(loaded_model.predict(image_arr))
        # get the classes from predicted probabilities
        print(img_probas)
        top3_idx = np.argsort(img_probas)[0][-3:]

        # create list to store the predictions
        predictions = []
        for idx in top3_idx[::-1]:
            img_cls = classes[idx]
            predictions.append([round(img_probas[0][idx], 2), img_cls])

        return predictions
    except Exception as e:
        print(f"Error predicting image: {e}")
        return []


# Define the Streamlit app
def app():
    st.title("Vehicle Type Image Classifier")

    # Add a file uploader
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    predictions = []
    # If an image is uploaded, make a prediction on it
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        predictions = get_predictions(image)
    else:
        st.write("No Image choosen")


    # Show the top 3 predictions with their probabilities
    if len(predictions):
        st.write("Top 3 class predictions:")
        for i, (proba, class_name) in enumerate(predictions):
            st.write(f"{i+1}. {class_name} ({proba*100:.2f}%)")
            progress_bar = st.progress(0)
            progress_bar.progress(int(proba*100))
    else:
        st.write("No predictions.")


# Run the app
if __name__ == "__main__":
    app()
