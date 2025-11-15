# 🧠 Dress vs Sneaker Classifier – Mindsight Technical Challenge

This project implements a simple binary image classifier for distinguishing between **Dress** and **Sneaker** items from the Fashion-MNIST dataset.  
The goal is to demonstrate clean code structure, fast prototyping, and working ML inference.

---

## 🚀 Features

- Custom Convolutional Neural Network (PyTorch)
- Training on filtered Fashion-MNIST dataset (2 classes only)
- CLI inference script for real images (PNG/JPG)
- Very small, clean, and modular codebase

---

## 📁 Project Structure

```
mindsight-challenge/
│
├── saved_models/
│   └── fashion_mnist_cnn.pth        # Trained model
│
├── src/
│   ├── model.py                     # CNN architecture
│   ├── train.py                     # Training script
│   ├── infer.py                     # CLI inference tool
│   └── utils.py                     # Helper functions
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sofiaaabrunori/mindsight-challenge.git
cd mindsight-challenge
```

Install dependencies:

```bash
pip3 install -r requirements.txt
```

---

## 🏋️ Training

Train the CNN:

```bash
python3 src/train.py
```

Expected output:

```
Train samples: 12000
Test samples: 2000
Test Accuracy: ~99%
Model saved to saved_models/fashion_mnist_cnn.pth
```

---

## 🔍 Inference (CLI)

Classify any local image:

```bash
python3 src/infer.py path_to_image.png
```

Example output:

```
Prediction: Sneaker
```

---

## 🧠 Model

- 2 convolutional layers  
- MaxPooling  
- 2 fully connected layers  
- Input resized to **28×28 grayscale** (Fashion-MNIST format)

---

## 📬 Contact

If you have any questions, feel free to reach out.
