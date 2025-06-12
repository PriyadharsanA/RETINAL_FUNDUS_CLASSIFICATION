import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="models/retinal_fundus.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

st.title("Fundus Disease Classifier (EfficientNet-B3 - TFLite)")

uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        predicted_class = predict_tflite_image(image, interpreter)
        st.success(f"Predicted Class: {predicted_class}")

# Set your image size (EfficientNetB3 default input)
IMAGE_SIZE = 380

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)  # Shape: (1, 380, 380, 3)

def predict_tflite_image(image: Image.Image, interpreter) -> int:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    return predicted_class
