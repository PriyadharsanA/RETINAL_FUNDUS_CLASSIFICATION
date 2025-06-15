import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = ['Diabetic Retinopathy', 'AMD', 'Glaucoma', 'Hypertensive Retinopathy']

# Preprocessing
def preprocess_image(image_file):
    image = Image.open(image_file).convert('RGB')
    image = image.resize((224, 224))  # Change size if needed
    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, image

# Streamlit app
st.set_page_config(page_title="TFLite Eye Disease Classifier")
st.title("🧠 Eye Disease Classification (TFLite)")
st.write("Upload a fundus image to classify eye diseases using a `.tflite` model.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img_array, pil_image = preprocess_image(uploaded_file)

    # Set tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    # Show results
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]

    st.markdown(f"### 🩺 Predicted Class: `{class_labels[class_idx]}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")

    # Optionally show full class probabilities
    st.write("Class probabilities:", dict(zip(class_labels, prediction.round(3))))
