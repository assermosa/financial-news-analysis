# 📈 Financial News Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-API-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/BERT-NLP-yellow?style=for-the-badge"/>
</p>

<p align="center">
  A multi-task BERT model for <strong>sentiment analysis</strong> and <strong>financial importance ranking</strong> of news articles — served via a Flask REST API and tracked with MLflow.
</p>

---

## 🧠 Overview

This project fine-tunes a **BERT-based model** on financial news to simultaneously tackle two tasks:

| Task | Description |
|------|-------------|
| 📊 **Sentiment Analysis** | Classifies news as Positive / Negative / Neutral |
| 🏆 **Importance Ranking** | Scores how financially significant a news article is |

By sharing a BERT encoder across both tasks, the model learns richer financial language representations while being efficient to serve.

---

## 🏗️ Architecture

```
Input News Article
        │
        ▼
  [BERT Encoder]  ←─ Shared backbone (bert-base-uncased / FinBERT)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
Sentiment  Importance
  Head      Head
   │         │
   ▼         ▼
Positive/ Score (0–1)
Negative/
Neutral
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/assermosa/financial-news-analysis.git
cd financial-news-analysis

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python train.py \
  --data_path data/financial_news.csv \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 2e-5
```

MLflow will automatically log metrics, parameters, and model artifacts during training.

```bash
# Launch MLflow UI to monitor experiments
mlflow ui
# Open http://localhost:5000 in your browser
```

---

## 🌐 Running the API

```bash
# Start the Flask server
python app.py
```

The API runs on `http://localhost:5000` by default.

### Endpoints

#### `POST /predict`

Analyze a news article for sentiment and importance.

**Request:**
```json
{
  "text": "Apple reports record-breaking Q4 earnings, beating analyst expectations by 15%."
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "sentiment_confidence": 0.94,
  "importance_score": 0.87,
  "importance_label": "high"
}
```

#### `GET /health`

```json
{ "status": "ok" }
```

---

## 📊 Results

> Best checkpoint restored at **Epoch 2** via Early Stopping (patience=3).

### Sentiment Analysis (Binary Classification)

| Split | Accuracy | Loss (BCE) |
|-------|----------|------------|
| Train | 83.26%   | 0.3767     |
| Val   | **84.83%**   | 0.3439     |

### Importance Ranking (Regression, scale −10 → +10)

| Split | MAE   | Loss (MSE) |
|-------|-------|------------|
| Train | 2.20  | 10.63      |
| Val   | **2.05**  | 10.60      |

### Training Config

| Hyperparameter | Value |
|----------------|-------|
| BERT backbone | `bert-base-uncased` |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 64 |
| Max Epochs | 8 (stopped at 2) |
| Dropout | 0.2 |
| Max Seq Length | 128 |

---

## 🛠️ Tech Stack

- **[BERT](https://arxiv.org/abs/1810.04805)** — Transformer backbone for NLP
- **[TensorFlow](https://www.tensorflow.org/)** — Model training & inference
- **[Flask](https://flask.palletsprojects.com/)** — Lightweight REST API
- **[MLflow](https://mlflow.org/)** — Experiment tracking & model registry
- **[Hugging Face Transformers](https://huggingface.co/transformers/)** — Pre-trained BERT weights

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ by <a href="https://github.com/assermosa">assermosa</a></p>
