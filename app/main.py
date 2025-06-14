from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = load_model("app/eye_disease_model.h5")
class_names = ['Normal','Mild DR','Moderate DR','Possible Glaucoma','Optic atrophy','Branch Retinal Vein Occlusion','Retinal Artery Occlusion','Rhegmatogenous Retinal Detachment','Maculopathy','Epiretinal Membrane','Macular Hole','Pathological Myopia'] 
IMG_SIZE = 380

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    pred_index = np.argmax(predictions[0])
    pred_class = class_names[pred_index]
    confidence = float(np.max(predictions[0]))

    return {"prediction": pred_class, "confidence": confidence}
