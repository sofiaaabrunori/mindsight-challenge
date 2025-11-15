import torch
from torchvision import datasets, transforms
from PIL import Image
import os

# -----------------------------
# Load FashionMNIST test set
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])
test_set = datasets.FashionMNIST(root="./", train=False, download=True, transform=transform)

classes = test_set.classes
SNEAKER = classes.index("Sneaker")
DRESS = classes.index("Dress")

# -----------------------------
# Create output directory
# -----------------------------
save_dir = "dataset_samples_for_inference"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# Helper function to save N samples of a class
# -----------------------------
def save_samples(target_label, n, prefix):
    count = 0
    for i in range(len(test_set)):
        img, label = test_set[i]
        if label == target_label:
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(os.path.join(save_dir, f"{prefix}_{count}.png"))
            count += 1
            if count == n:
                break

# -----------------------------
# Save 2 Sneaker and 2 Dress
# -----------------------------
save_samples(SNEAKER, 2, "sneaker")
save_samples(DRESS, 2, "dress")

print("✅ Saved images in:", save_dir)
print("Files created:")
print(os.listdir(save_dir))
