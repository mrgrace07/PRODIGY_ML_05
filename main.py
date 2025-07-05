from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import cv2
import numpy as np
from skimage.feature import hog
import pickle
import pandas as pd
from uuid import uuid4
import uvicorn
from fastapi.staticfiles import StaticFiles



app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (100, 100)



# Global model and data
model = None
nutrition_df = None

def load_model_and_data():
    global model, nutrition_df
    with open("food_model.pkl", "rb") as f:
        model = pickle.load(f)
    nutrition_df = pd.read_csv("harrish-nutrition.csv")
    print("✅ Model and nutrition data loaded.")

load_model_and_data()

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"❌ Unable to read image at path: {image_path}")

    img = cv2.resize(img, IMG_SIZE)
    features, _ = hog(img, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      orientations=9,
                      block_norm='L2-Hys',
                      visualize=True)
    return features

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(None), dish_name: str = Form(None)):
    try:
        prediction = None

        # ✅ IMAGE INPUT
        if image and image.filename:
            file_ext = os.path.splitext(image.filename)[1]
            file_path = os.path.join(UPLOAD_FOLDER, f"{uuid4().hex}{file_ext}")
            with open(file_path, "wb") as f:
                f.write(await image.read())

            # Check if image is valid
            features = extract_features(file_path)
            prediction = model.predict([features])[0]

        # ✅ TEXT INPUT
        elif dish_name and dish_name.strip() != "":
            prediction = dish_name.strip()

        # ❌ No input at all
        else:
            raise ValueError("Please upload an image or enter a dish name.")

        # ✅ Get nutrition info
        nutrition = nutrition_df[nutrition_df['Dish Name'].str.strip().str.lower().str.contains(prediction.lower())]
        nutrition_data = nutrition.to_dict(orient="records") if not nutrition.empty else []
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {
                "prediction": prediction,
                "nutrition": nutrition_data if nutrition_data else {"Message": "No nutrition info found"}
            }
        })

    except Exception as e:
        print("❌ Error during prediction:", e)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"prediction": "Error", "nutrition": {"Error": str(e)}}
        })
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1",port=5000)
