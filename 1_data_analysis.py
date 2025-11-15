import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Load the FashionMNIST dataset
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.FashionMNIST(
    root="./", train=True, download=True, transform=transform
)

test_set = datasets.FashionMNIST(
    root="./", train=False, download=True, transform=transform
)

classes = train_set.classes


# -----------------------------
# 2. Print dataset statistics
# -----------------------------
print("📊 DATASET INFO")
print("---------------------")
print(f"Training images: {len(train_set)}")
print(f"Test images: {len(test_set)}")
print(f"Number of classes: {len(classes)}\n")

print("📌 Class names:")
for i, c in enumerate(classes):
    print(f"{i}: {c}")

# Count images per class (train)
class_counts = {c: 0 for c in classes}

for _, label in train_set:
    class_counts[classes[label]] += 1

print("\n📊 Images per class (training set):")
for c, count in class_counts.items():
    print(f"{c}: {count}")


# -----------------------------
# 3. Function to display samples
# -----------------------------
def show_examples(dataset, target_class, n=5):
    """Show n examples of a specific class."""
    idxs = [i for i, (_, label) in enumerate(dataset) if label == target_class]

    plt.figure(figsize=(10, 2))
    plt.suptitle(f"Examples of class: {classes[target_class]}", fontsize=16)

    for i in range(n):
        img, _ = dataset[idxs[i]]
        img = img.squeeze(0)

        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.show()


# -----------------------------
# 4. Visualize Sneakers and Dress
# -----------------------------
print("\n🔍 Showing examples of Sneaker and Dress...")
SNEAKER = classes.index("Sneaker")
DRESS = classes.index("Dress")

show_examples(train_set, SNEAKER, n=5)
show_examples(train_set, DRESS, n=5)

print("\n✅ Data analysis completed.")
