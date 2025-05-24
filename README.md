**# 💳 Real-Time Credit Card Fraud Detection System using Python and Machine Learning

This project is a real-time system designed to detect fraudulent credit card transactions using machine learning. It simulates live data,
detects anomalies using the Isolation Forest algorithm, and displays results on an interactive dashboard with Streamlit.


## 🚀 Features

- Real-time credit card transaction simulation
- Anomaly detection using Isolation Forest
- Interactive Streamlit dashboard
- Model and scaler saved using `joblib`
- Easy to understand and extend

---

## 🧠 Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy`
  - `scikit-learn` (ML)
  - `joblib` (model persistence)
  - `streamlit` (UI dashboard)
  - `faker` (data simulation)

---

## 📂 Project Structure

```bash
fraud_detection_project/
├── app/
│   ├── main.py              # Streamlit dashboard for real-time fraud detection
│   ├── train_model.py       # Script to train Isolation Forest model
│   ├── data/
│   │   └── creditcard.csv   # Source dataset from Kaggle
│   ├── model/
│       ├── fraud_model.pkl  # Trained Isolation Forest model
│       └── scaler.pkl       # StandardScaler used during preprocessing
```

---

## 🧪 How It Works

### 1. Train the Model

Run the training script:

```bash
cd app
python train_model.py
```

This will:
- Load and preprocess the dataset.
- Train the Isolation Forest model on selected features.
- Save the model and scaler as `.pkl` files.

### 2. Launch the Real-Time App

Start the Streamlit dashboard:

```bash
streamlit run main.py
```

This will open a live dashboard that:
- Simulates transactions every second.
- Classifies them as **fraudulent** or **legitimate**.
- Shows stats and transaction logs in real time.

---

## 📊 Model Details

- **Algorithm**: Isolation Forest (unsupervised)
- **Input Features**: `Amount`, `Hour`
- **Output**: Binary classification (Fraud / Not Fraud)
- **Why Isolation Forest?**
  - Works well for anomaly detection without labeled data.
  - Fast and scalable for real-time inference.

---

## ✅ Advantages

- Real-time simulation with UI
- Easy to deploy and extend
- Lightweight and scalable model
- End-to-end ML lifecycle: training → prediction → visualization

---

## 📈 Future Scope

- Use the full 29 features from the dataset for better accuracy
- Integrate with real transaction APIs (e.g., Kafka, WebSockets)
- Add alerting system (email/SMS) for detected frauds
- Include login panel and admin dashboard

---

## 📁 Dataset

Credit card transactions dataset sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 👨‍💻 Author

**Anil Kumar Tholleti**  
[GitHub Profile](https://github.com/Anil-Kumar-Tholleti)

---

## 📜 License

This project is open-source and free to use for educational and non-commercial purposes.
**
