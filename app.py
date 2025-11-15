import io
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from PIL import Image

from src.model import SimpleCNN
from src.utils import preprocess_image

app = FastAPI(title="Dress vs Sneaker Classifier API")

# Load model
MODEL_PATH = "saved_models/fashion_mnist_cnn.pth"
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

CLASS_NAMES = ["Dress", "Sneaker"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("L")

    # Preprocess image (PIL → tensor)
    tensor = preprocess_image(img)

    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)

    return {"prediction": CLASS_NAMES[predicted.item()]}


@app.get("/")
def home():
    return {"message": "Dress vs Sneaker Classification API is running!"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

