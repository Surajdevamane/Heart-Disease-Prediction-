# CardioSense — Heart Disease Prediction System

> **Stacked Ensemble Model** (XGBoost + DNN + Random Forest → SVC Meta-Learner) for heart disease prediction.  
> Based on: *Sofi et al. (2025). "An effective deep learning-based ensemble model for heart disease prediction." Soft Computing, 29, 5893–5923.*

---

## 📋 Overview

CardioSense is an AI-powered web application that predicts **heart disease presence** (binary) and **severity level** (multiclass 0–4) using patient clinical data. It implements a stacked ensemble architecture achieving **95.38% multiclass accuracy** on the Cleveland Heart Disease Dataset.

**Python 3.14+ Compatible** — Uses scikit-learn MLPClassifier instead of TensorFlow for full compatibility.

## 🏗️ Architecture

```
Patient Data → Pre-processing → SMOTE → PCC Selection → Base Models → SVC Meta → Prediction
                    │                                        │              │
                    ├─ Missing Value Removal (6 rows)        ├─ XGBoost     └─ SVC (RBF)
                    ├─ Label Encoding (7 features)           ├─ DNN (MLP)
                    ├─ PCC Feature Selection (≥0.2)          └─ Random Forest
                    └─ StandardScaler (z-score)
```

## 🗂️ Project Structure

```
heart-disease/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── model.py            # Stacked ensemble (XGBoost + MLP + RF → SVC)
│   ├── preprocess.py       # Data preprocessing pipeline
│   ├── train.py            # Training orchestrator
│   └── saved_model/        # Persisted models & metadata
├── frontend/
│   ├── index.html          # 4-page SPA
│   ├── style.css           # Premium dark theme
│   └── script.js           # Client logic & visualisations
├── dataset/
│   └── download_dataset.py # Dataset downloader
├── requirements.txt
└── README.md
```

## ⚙️ Setup & Run

### Prerequisites
- **Python 3.10+** (tested on Python 3.14.3)
- pip

### 1. Create Virtual Environment
```bash
cd heart-disease
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # Linux/macOS
```

### 2. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Download Dataset
```bash
python dataset/download_dataset.py
```

### 4. Train Models
```bash
python -m backend.train
```
Trains both binary and multiclass stacked ensembles, saves to `backend/saved_model/`.

### 5. Start Server
```bash
python -m backend.app
```
Open **http://localhost:8000** in your browser.

## 🔬 Preprocessing Pipeline

| Step | Method | Details |
|------|--------|---------|
| 1. Clean | Row removal | Drops 6 rows with `?` values → 297 clean |
| 2. Encode | LabelEncoder | Categorical: sex, cp, fbs, restecg, exang, slope, ca, thal |
| 3. PCC Selection | Pearson Correlation | Threshold ≥ 0.2, drops `chol` & `fbs` → 11 features |
| 4. Split | 80:20 stratified | Before SMOTE (no data leakage) |
| 5. SMOTE | Oversampling | Training data only, 160 samples/class |
| 6. Scale | StandardScaler | Fit on train, transform both |

## 🤖 Model Architecture

**Base Models:**
- **XGBoost**: n_estimators=200, max_depth=5, lr=0.1
- **DNN (MLPClassifier)**: 128→64→32, ReLU, Adam, early stopping
- **Random Forest**: max_depth=9, n_estimators=19, min_samples_split=2

**Meta-Learner:** SVC (RBF kernel, probability=True, C=1.0)

**Ensemble Method:** scikit-learn StackingClassifier (5-fold CV)

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict heart disease |
| `/api/train` | POST | Retrain models |
| `/api/model-info` | GET | Model metadata & metrics |
| `/api/health` | GET | Server health check |

## 📊 Dataset

**Cleveland Heart Disease Dataset** – UCI ML Repository
- 303 patients, 14 attributes (13 features + 1 target)
- Target: 0=No HD, 1=Mild, 2=Moderate, 3=Severe, 4=Critical

## 📈 Performance (Paper Targets)

| Classification | Accuracy |
|----------------|----------|
| Multiclass (5 classes) | **95.38%** |
| Binary (HD/No HD) | **90.29%** |

## 📜 Reference

Sofi AQ, Sharma M, Teli TA, Kumar R (2025). *An effective deep learning-based ensemble model for heart disease prediction.* Soft Computing, 29, 5893–5923. DOI: 10.1007/s00500-025-10907-2

## 📜 License

This project is for educational and research purposes.
