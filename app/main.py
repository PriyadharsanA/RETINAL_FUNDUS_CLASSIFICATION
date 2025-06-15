import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/retinal_fundus (1).tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = ['0.0.Normal','0.3.DR1','1.0.DR2','10.0.Possible glaucoma','10.1.Optic atrophy','2.0.BRVO','3.RAO','4.Rhegmatogenous RD', '6.Maculopathy','7.ERM','8.MH','9.Pathological myopia']

# Preprocessing
def preprocess_image(image_file):
    image = Image.open(image_file).convert('RGB')
    image = image.resize((380,380))  # Change size if needed
    img = np.array(image, dtype=np.float32)
    img = preprocess_input(image)
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
    st.write("Expected shape:", input_details[0]['shape'])
    st.write("Expected dtype:", input_details[0]['dtype'])
    st.write("Your image shape:", img_array.shape)
    st.write("Your image dtype:", img_array.dtype)
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
