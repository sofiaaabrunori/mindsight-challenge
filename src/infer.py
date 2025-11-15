import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SimpleCNN
import sys

CLASS_NAMES = ["Dress", "Sneaker"]

def load_model(model_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0)  # batch dimension

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return CLASS_NAMES[predicted.item()]

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 infer.py <image_path>")
        return

    image_path = sys.argv[1]
    model = load_model("../saved_models/fashion_mnist_cnn.pth")
    prediction = predict(image_path, model)

    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
