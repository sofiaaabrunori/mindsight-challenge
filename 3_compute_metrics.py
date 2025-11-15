import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.model import SimpleCNN


# -------------------------------
# 1) Same transforms as training
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

TARGET_CLASSES = [3, 7]  # Dress, Sneaker


# -------------------------------
# 2) Load full FashionMNIST TEST
# -------------------------------
test_full = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Filter only Dress (3) and Sneaker (7)
indices = [i for i, (_, label) in enumerate(test_full) if label in TARGET_CLASSES]
test_subset = Subset(test_full, indices)

test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)


# -------------------------------
# 3) Load trained model
# -------------------------------
MODEL_PATH = "saved_models/fashion_mnist_cnn.pth"

model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


# -------------------------------
# 4) Evaluate on filtered test set
# -------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:

        # Remap labels: 3→0 (Dress), 7→1 (Sneaker)
        mapped = torch.where(labels == 3, torch.tensor(0), torch.tensor(1))

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.numpy())
        all_labels.extend(mapped.numpy())


# -------------------------------
# 5) Convert numbers → names
# -------------------------------
idx_to_name = {0: "Dress", 1: "Sneaker"}

true_names = [idx_to_name[x] for x in all_labels]
pred_names = [idx_to_name[x] for x in all_preds]


# -------------------------------
# 6) Print metrics
# -------------------------------
print("\n📊 Classification Report (Dress vs Sneaker)")
report = classification_report(
    true_names, pred_names,
    target_names=["Dress", "Sneaker"]
)
print(report)

with open("metrics_report_binary.txt", "w") as f:
    f.write(report)

# -------------------------------
# 7) Confusion Matrix (2×2)
# -------------------------------
cm = confusion_matrix(true_names, pred_names, labels=["Dress", "Sneaker"])
print("\n🔢 Confusion Matrix:")
print(cm)
