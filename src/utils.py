import torch
from torchvision import transforms
from PIL import Image

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(img).unsqueeze(0)  # add batch dimension
    return tensor
