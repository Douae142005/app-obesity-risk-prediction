# 🏥 Obesity Risk Estimation — Medical Decision Support Application
> **Coding Week · 09–15 March 2026 · École Centrale Casablanca**  
> An explainable machine learning tool to help physicians estimate patient obesity risk based on lifestyle and physical conditions.
![CI](https://github.com/Douae142005/app-obesity-risk-prediction/actions/workflows/ci.yml/badge.svg)
---

## I. Project Overview

This clinical decision-support tool predicts a patient's obesity level (7 classes) from lifestyle and physical data. The solution prioritizes:

- **Accuracy** — multi-model comparison with LightGBM selected as final model
- **Interpretability** — SHAP explanations for every prediction
- **Usability** — Streamlit interface designed for physicians
- **Reproducibility** — fully automated pipeline from data loading to deployment

**Dataset:** [UCI ML Repository — Estimation of Obesity Levels](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)  
**Target:** `NObeyesdad` — 7 obesity classes from `Insufficient_Weight` to `Obesity_Type_III`  
**Best model:** LightGBM (~96% accuracy, F1-Weighted ~0.96, ROC-AUC > 0.99)

---

## II. Project Structure

```
app-obesity-risk-prediction/
│
└── project/
    │
    ├── app/
    │   └── app.py
    │
    ├── data/
    │   ├── test_processed.csv
    │   └── train_processed.csv
    │
    ├── docs/
    │   └── prompt_engineering.md
    │
    ├── models/
    │   ├── best_model.pkl
    │   ├── confusion_matrix_lightgbm.png
    │   ├── confusion_matrix_random_forest.png
    │   ├── confusion_matrix_xgboost.png
    │   ├── label_encoders.pkl
    │   └── scaler.pkl
    │
    ├── notebooks/
    │   └── eda.ipynb
    │
    ├── outputs/
    │   ├── confusion_matrix_final.png
    │   └── roc_curves.png
    │
    ├── src/
    │   ├── __pycache__/
    │   ├── data_processing.py
    │   ├── evaluate_model.py
    │   └── train_model.py
    │
    ├── tests/
    │   ├── __pycache__/
    │   ├── __init__.py
    │   ├── test_data_processing.py
    │   ├── test_evaluate_model.py
    │   ├── test_memory_optimization.py
    │   └── test_model.py
    │
    ├── Dockerfile
    ├── README.md
    ├── requirements.txt
    └── .gitignore
```

---

### II.1 — `app/app.py` — Streamlit Interface

The web interface built with Streamlit allows physicians to interact with the model without any technical knowledge.
![App Screenshot](docs/screenshot.png)

**Key points:**
- Input form covering all 16 patient features (age, weight, height, diet, activity, etc.)
- Loads `best_model.pkl`, `label_encoders.pkl`, and `scaler.pkl` from `models/` at startup
- Displays the predicted obesity class with its confidence score via `predict_proba`
- Renders a SHAP force plot for each individual prediction
- Color-coded risk indicator for quick clinical reading

---

### II.2 — `data/` — Processed Data

Contains the preprocessed datasets produced by `save_processed_data()` in `data_processing.py`.

| File | Description |
|---|---|
| `train_processed.csv` | Encoded and normalized training set (80%) |
| `test_processed.csv` | Encoded and normalized test set (20%) |

These files allow offline reuse without re-fetching the UCI dataset on subsequent runs. They are regenerated automatically if `train_model.py` is re-executed.

---

### II.3 — `docs/prompt_engineering.md` — Prompt Engineering Documentation

Dedicated markdown file documenting the AI-assisted development workflow across the full project.

**AI tools used:**

| AI Tool | Type | Primary Usage |
|---|---|---|
| **Claude (Anthropic)** | Conversational assistant | Data processing, code explanation |
| **GitHub Copilot** | IDE completion | EDA, Streamlit interface |
| **ChatGPT (GPT-4)** | Conversational assistant | ML models, tests, CI/CD |
| **DeepSeek Coder** | Code assistant | Optimization, debugging |
| **OpenAI Codex** | Code generation | Automation, scripts |

**Contains:**
- Exact prompts used per tool and per task, with corresponding outputs
- What was kept as-is vs. manually adjusted after generation
- Analysis of each tool's effectiveness for its assigned task
- Lessons learned and suggested prompt improvements for future iterations

---

### II.4 — `models/` — Saved Artifacts

Auto-generated folder, populated after running `train_model.py`.

| File | Description |
|---|---|
| `best_model.pkl` | Serialized LightGBM model — best across the 3 trained candidates |
| `label_encoders.pkl` | `LabelEncoder` objects for all categorical columns |
| `scaler.pkl` | `StandardScaler` fitted on `X_train` only (prevents data leakage) |
| `confusion_matrix_lightgbm.png` | Per-model confusion matrix — LightGBM |
| `confusion_matrix_random_forest.png` | Per-model confusion matrix — Random Forest |
| `confusion_matrix_xgboost.png` | Per-model confusion matrix — XGBoost |

**Critical design note:** the scaler is fitted exclusively on `X_train` inside `normalize_features()`. Only `.transform()` is applied to `X_test` — this prevents any test-set information from leaking into the training process.

---

### II.5 — `notebooks/eda.ipynb` — Exploratory Data Analysis

Full pre-training analysis documenting all key decisions taken before modeling.

**Missing values** — `df.isnull().sum().sum() == 0`. No missing values found. `handle_missing_values()` is kept as a production safeguard (numeric → median, categorical → mode).

**Memory profiling** — `df.info()` revealed all numerical columns as `float64`/`int64`, totaling ~280.5 KB. Decision: apply `optimize_memory()` early in the pipeline to halve memory usage before any further processing.

**Class distribution** — All 7 obesity classes fall between 12% and 15%. Dataset is balanced. Decision: use `stratify=y` in the split — no SMOTE or class weighting needed.

**Outliers** — Boxplots on 8 numerical features (`Age`, `Height`, `Weight`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`). All extreme values are medically plausible (e.g., Weight = 170 kg, Age = 61). Decision: no outlier removal.

**Correlations** — No feature pair exceeded 0.8 Pearson correlation. Highest: Height ↔ Weight (0.46), Age ↔ TUE (−0.30). Decision: all features retained, no dimensionality reduction applied.

---

### II.6 — `outputs/` — Evaluation Plots

Auto-generated folder, populated after running `evaluate_model.py`.

| File | Description |
|---|---|
| `confusion_matrix_final.png` | Confusion matrix of the best model on the full test set |
| `roc_curves.png` | One ROC curve per class (One-vs-Rest) with individual AUC scores |

---

### II.7 — `src/data_processing.py` — Preprocessing Pipeline

Central module called by both `train_model.py` and `evaluate_model.py`. Ensures fully consistent preprocessing across training and inference.

**Full pipeline (`preprocess_pipeline()`):**

```
load_data()              → fetch dataset from UCI (id=544)
handle_missing_values()  → safeguard fill (median / mode)
optimize_memory()        → float64→float32, int64→int32
encode_features()        → LabelEncoder on all object columns
split_features_target()  → separate X and y on NObeyesdad
split_train_test()       → 80/20 stratified split, random_state=42
normalize_features()     → StandardScaler fit on X_train only
save_processed_data()    → CSV + encoders + scaler to disk
```

**Key function — `optimize_memory(df)`:**
```python
def optimize_memory(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df
```
Memory before: ~280.5 KB → after: ~140.3 KB → **~50% reduction ✅**

---

### II.8 — `src/train_model.py` — Model Training & Comparison

Trains 3 models sequentially, evaluates each on the test set, selects the best by accuracy, and saves it.

**Models trained:**

| Model | Accuracy | F1-Score (Weighted) |
|---|---|---|
| Random Forest | ~0.94 | ~0.94 |
| XGBoost | ~0.95 | ~0.95 |
| **LightGBM ✅** | **~0.96** | **~0.96** |

**Key design points:**
- Each model evaluated with `accuracy_score` and `f1_score(average='weighted')` on `X_test`
- `select_best_model()` picks the model with the highest accuracy
- Individual confusion matrix saved per model in `models/`
- `save_best_model()` serializes the winner via `joblib.dump()` to `models/best_model.pkl`
- `random_state=42` set across all models and splits for full reproducibility

---

### II.9 — `src/evaluate_model.py` — Deep Evaluation

Loads `best_model.pkl` and runs a complete diagnostic on the test set. Must be run after `train_model.py`.

**Metrics computed:**
- Accuracy, F1-Macro, F1-Weighted
- ROC-AUC macro (One-vs-Rest via `label_binarize`)
- Full `classification_report` with per-class precision / recall / F1

**Clinical interpretation block** — automatic verdict printed to console:
- 🟢 ROC-AUC ≥ 0.95 → reliable for clinical decision support
- 🟡 0.85–0.95 → medical supervision recommended
- 🔴 < 0.85 → not suitable for production

---

### II.10 — `tests/` — Automated Tests

4 test files executed automatically via GitHub Actions on every push and pull request.

| File | What it verifies |
|---|---|
| `test_data_processing.py` | No NaN after `handle_missing_values()`, correct pipeline output shapes |
| `test_evaluate_model.py` | Metrics computation runs without error, output values in valid range |
| `test_memory_optimization.py` | `optimize_memory()` reduces memory usage and produces `float32`/`int32` dtypes |
| `test_model.py` | `best_model.pkl` loads correctly and returns valid class predictions |

`__init__.py` is included to make `tests/` a proper Python package, enabling pytest discovery across all files.

---

### II.11 — `Dockerfile` — Containerization

Packages the full application for portable, environment-independent deployment.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY project/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY project/ .
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

### II.12 — `requirements.txt` — Dependencies

```
pandas
numpy
scikit-learn
lightgbm
xgboost
shap
streamlit
matplotlib
seaborn
joblib
ucimlrepo
pytest
```

---

## III. Installation & Usage

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-team>/app-obesity-risk-prediction.git
cd app-obesity-risk-prediction/project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Downloads the dataset from UCI, runs the full preprocessing pipeline, trains 3 models, selects the best, and saves all artifacts to `models/`.

```bash
python src/train_model.py
```

### 4. (Optional) Run full evaluation

Generates confusion matrix and ROC curves in `outputs/`.

```bash
python src/evaluate_model.py
```

### 5. Launch the Streamlit application

```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501`.

### 6. (Optional) Run with Docker

```bash
docker build -t obesity-risk-app .
docker run -p 8501:8501 obesity-risk-app
```

### 7. Run automated tests

```bash
pytest tests/
```

---

## IV. Critical Questions

**Was the dataset balanced? If not, how was imbalance handled?**  
Yes — all 7 classes are distributed between ~12% and ~15% per class. The dataset is effectively balanced. `stratify=y` was used in the train/test split to preserve this distribution. No SMOTE, undersampling, or class weighting was applied. F1-Macro was used as a secondary metric to ensure equal attention to all classes regardless of minor size variations. Impact: no class was systematically under-predicted.

---

**Which ML model performed best? Performance metrics?**  
**LightGBM** achieved the best results across all metrics:

| Metric | Score |
|---|---|
| Accuracy | ~0.96 |
| F1-Weighted | ~0.96 |
| F1-Macro | ~0.95 |
| ROC-AUC (macro OvR) | > 0.99 |

LightGBM was selected for its speed advantage over XGBoost on this dataset size, its native SHAP compatibility via `TreeExplainer`, and its consistent superiority across all evaluation runs.

---

**Which medical features most influenced predictions (SHAP results)?**  
Based on SHAP summary plots, the top drivers are:

1. **Weight** — strongest single predictor across all obesity classes
2. **Height** — interacts with weight (implicit BMI relationship)
3. **FAF** (Physical Activity Frequency) — low activity strongly pushes toward higher obesity classes
4. **FCVC** (Vegetable Consumption Frequency) — lower consumption linked to higher risk
5. **Age** — moderate influence, especially for `Obesity_Type_III`
6. **NCP** (Number of main meals per day) — eating pattern signal
7. **CH2O** (Daily water intake) — hydration correlates with healthier weight profiles

These findings are clinically coherent and reinforce physician trust in the model's predictions.

---

**What insights did prompt engineering provide?**  
Full documentation is in [`docs/prompt_engineering.md`](docs/prompt_engineering.md). Summary:

5 AI tools were used across the project, each assigned to the tasks that best matched their strengths:

| AI Tool | Task | Key insight |
|---|---|---|
| **Claude** | Data processing, code explanation | Best for understanding and documenting complex pipeline logic |
| **GitHub Copilot** | EDA, Streamlit interface | Fastest for in-editor boilerplate and repetitive UI patterns |
| **ChatGPT (GPT-4)** | ML models, tests, CI/CD | Reliable for structured multi-step tasks like GitHub Actions config |
| **DeepSeek Coder** | Optimization, debugging | Effective for targeted code fixes with minimal context |
| **OpenAI Codex** | Automation, scripts | Efficient for generating standalone utility scripts |

Key lesson: **specificity in prompts directly reduces iteration cycles** — providing function names, expected input/output types, and desired output format in the initial prompt consistently produced working code in one shot. Vague prompts required 2–3 follow-up exchanges to reach the same result.

---
## 🚀 Quick Start

### 1. Cloner le repo
```bash
git clone https://github.com/Douae142005/app-obesity-risk-prediction.git
cd app-obesity-risk-prediction/project
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Entraîner le modèle
```bash
python src/data_processing.py
python src/train_model.py
python src/evaluate_model.py
```

### 4. Lancer l'application
```bash
streamlit run app/app.py
```

### 5. Lancer les tests
```bash
pytest tests/ -v
```
*École Centrale Casablanca · Coding Week March 2026 · Supervised by soufiane mehdi. team Data Healers: amghar douae, azoud hajar, boutalmaouine amina, dyaz hajar, querchi meryem*


