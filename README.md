# 🚀 HashBERT: Rolling Hash-Based Lightweight Transformer

**HashBERT** is a lightweight BERT-inspired NLP model that leverages **dynamically generated rolling hash embeddings** to significantly reduce model size while maintaining high accuracy. It enables efficient **on-device deployment** of deep learning models on memory-constrained devices such as smartphones and smartwatches.

---

## 🧠 Project Highlights

- ✅ Over **93% reduction in model size** for CNN-based architectures.
- ✅ Maintains comparable performance to **Word2Vec** and **Google's pre-trained embeddings**.
- ✅ Enables **sentiment analysis** on large-scale datasets using compact deep learning models.
- ✅ Built for deployment on **resource-constrained edge devices**.

---

## 📦 Dataset

- **Dataset:** [Amazon Reviews Dataset](https://nijianmo.github.io/amazon/index.html)
- **Size:** 2,640,254 rows × 15 columns
- **Tasks:**
  - **Binary Sentiment Classification:** Positive / Negative
    - Train: 200K samples
    - Test: 40K samples
  - **Ternary Sentiment Classification:** Positive / Negative / Neutral
    - Train: 250K samples
    - Test: 50K samples

---

## 🧪 Experiment Setup

### 🏗️ Models Used
- **Feed Forward Neural Network (FFN)**
- **Convolutional Neural Network (CNN)**

### 📥 Embedding Methods:
- Google Word2Vec (pre-trained on Google News)
- Custom Word2Vec (via GenSim)
- Proposed Rolling Hash Embeddings

---

## 📊 Results

### 🔍 Accuracy Comparison

| Task             | Google Word2Vec | Custom Word2Vec | HashBERT (Before Tuning) | HashBERT (After Tuning) | Accuracy Change vs Google | Accuracy Change vs Word2Vec |
|------------------|------------------|------------------|----------------------------|---------------------------|----------------------------|------------------------------|
| **Binary FFN**   | 85.48%           | 87.58%           | 79.08%                     | 81.50%                    | -3.98%                     | -6.08%                       |
| **Ternary FFN**  | 69.98%           | 71.82%           | 63.05%                     | 68.51%                    | -1.46%                     | -3.30%                       |
| **Binary CNN**   | 81.62%           | 87.83%           | 83.10%                     | 87.68%                    | +6.07%                     | -0.15%                       |
| **Ternary CNN**  | 68.44%           | 72.88%           | 67.46%                     | 72.02%                    | +3.58%                     | -0.86%                       |

---

### 📦 Model Size Comparison

| Task             | Google / Word2Vec | HashBERT Model Size | Size Reduction vs Google | Size Reduction vs Word2Vec |
|------------------|-------------------|----------------------|---------------------------|-----------------------------|
| **Binary FFN**   | 124 KB            | 63.6 KB              | -48.70%                   | -48.70%                     |
| **Ternary FFN**  | 124 KB            | 63.7 KB              | -48.62%                   | -48.62%                     |
| **Binary CNN**   | 2.7 MB            | 185.4 KB             | -93.95%                   | -93.95%                     |
| **Ternary CNN**  | 2.7 MB            | 190.0 KB             | -92.90%                   | -92.90%                     |

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **NLP / Transformers**
- **Custom Rolling Hash Embedding**
- **GenSim for Word2Vec**
- **Scikit-learn**
- **Matplotlib / Seaborn**

---

## 📌 Key Contributions

- 🔧 Designed an **embedding-less approach** using **rolling hash algorithms** for compact vector representations.
- 🧩 Implemented efficient end-to-end **NLP pipelines** with PyTorch for both FFN and CNN models.
- 📉 Achieved **massive compression** (up to ~94%) enabling lightweight model deployment for **edge AI**.

---

## 📈 Future Work

- ⚙️ Integration with real-world mobile and embedded environments (e.g., TensorFlow Lite, ONNX).
- 🔄 Exploring alternate hashing techniques (e.g., MinHash, SimHash).
- 📡 Benchmarking inference times across various edge hardware.

---

## 📚 Citation & References

> Inspired by EELBERT (Embeddingless BERT) and advances in edge-device NLP.
