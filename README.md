# Mindsight Challenge – Dress vs Sneaker Classifier 👗👟

This project implements a complete **end-to-end machine learning pipeline** for binary image classification using **FashionMNIST**.  
The model distinguishes **Dress (label 3)** from **Sneaker (label 7)** using a custom CNN, an inference API, and evaluation metrics.

---

# 🚀 Project Structure

```
mindsight-challenge/
│── src/
│   │── train.py
│   │── model.py
│   │── infer.py
│   │── utils.py
│
│── saved_models/
│   └── fashion_mnist_cnn.pth
│
│── dataset_samples_for_inference/
│   ├── dress_0.png
│   ├── dress_1.png
│   ├── sneaker_0.png
│   ├── sneaker_1.png
│
│── 1_data_analysis.py
│── 2_prepare_inference_samples.py
│── 3_compute_metrics.py
│── app.py
│── requirements.txt
│── Dockerfile
│── README.md
```

---

# 📊 1. Dataset Analysis

Run:

```bash
python3 1_data_analysis.py
```

This script:

- Downloads **FashionMNIST**
- Prints dataset size
- Shows class distribution
- Visualizes sample images (train + test)
- Confirms that:  
  - Dress = **label 3**  
  - Sneaker = **label 7**

---

# 🎨 2. Training the Model (Binary Classifier)

Run:

```bash
python3 src/train.py
```

Training pipeline:

- Filters only classes **3 (Dress)** and **7 (Sneaker)**
- Remaps labels:
  - Dress → 0  
  - Sneaker → 1
- Trains a lightweight **CNN**
- Achieves > **99% accuracy**
- Saves trained model in:

```
saved_models/fashion_mnist_cnn.pth
```

---

# 🧪 3. Preparing Sample Images for Inference

Run:

```bash
python3 2_prepare_inference_samples.py
```

This script extracts **real test samples** from FashionMNIST and stores them inside:

```
dataset_samples_for_inference/
    dress_0.png
    dress_1.png
    sneaker_0.png
    sneaker_1.png
```

You can upload these directly into Swagger UI.

---

# 📈 4. Computing Metrics (Accuracy, F1-score)

Run:

```bash
python3 3_compute_metrics.py
```

This script evaluates the trained model using the filtered test set  
(Dress vs Sneaker only) and prints:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

A full report is also saved as:

```
metrics_report_binary.txt
```

---

# 🌐 5. REST API Inference (FastAPI + Swagger UI)

Start the API:

```bash
uvicorn app:app --reload
```

Then open:

👉 **http://127.0.0.1:8000/docs**

You can upload any image (PNG/JPG).  
The API will:

- Convert to grayscale  
- Resize to 28×28  
- Normalize  
- Predict **"Dress"** or **"Sneaker"**

---

# 💻 6. Terminal Inference

```bash
python3 src/infer.py dataset_samples_for_inference/sneaker_0.png
```

---

# 🐳 7. Docker Support

Build:

```bash
docker build -t mindsight-api .
```

Run:

```bash
docker run -p 8000:8000 mindsight-api
```

---

# 📦 Requirements

```
fastapi
uvicorn
torch
torchvision
pillow
numpy
python-multipart
scikit-learn
matplotlib
```

Install all with:

```bash
pip install -r requirements.txt
```

---

# ✨ Author
**Sofia Brunori** — Mindsight Challenge Submission
