import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from model import SimpleCNN
import os

BATCH_SIZE = 64
EPOCHS = 5
TARGET_CLASSES = [3, 7]  # Dress, Sneaker

class RemapDataset(Dataset):
    # Map labels: 3 → 0, 7 → 1
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        new_label = 0 if label == 3 else 1
        return img, new_label

def filter_classes(dataset, classes):
    idx = [i for i in range(len(dataset)) if dataset[i][1] in classes]
    return Subset(dataset, idx)

def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_full = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_full = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    train_subset = filter_classes(train_full, TARGET_CLASSES)
    test_subset = filter_classes(test_full, TARGET_CLASSES)

    # ✔ remap labels 3→0 , 7→1
    train_dataset = RemapDataset(train_subset)
    test_dataset = RemapDataset(test_subset)

    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # MODEL
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model (Dress vs Sneaker)...")

    # TRAINING LOOP
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    # TEST
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    os.makedirs("../saved_models", exist_ok=True)
    torch.save(model.state_dict(), "../saved_models/fashion_mnist_cnn.pth")

    print("Model saved to saved_models/fashion_mnist_cnn.pth")

if __name__ == "__main__":
    main()
