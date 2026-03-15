# 🏥 Obesity Risk Estimation — Medical Decision Support Application

> **Coding Week · 09–15 March 2026 · École Centrale Casablanca**  
> An explainable machine learning tool to help physicians estimate patient obesity risk based on lifestyle and physical conditions.

![CI](https://github.com/Douae142005/app-obesity-risk-prediction/actions/workflows/ci.yml/badge.svg)

---

## 📚 Table of Contents

- [I. Project Overview](#i-project-overview)
- [II. Project Structure](#ii-project-structure)
  - [II.1 — app/app.py](#ii1--appapppy--streamlit-interface)
  - [II.2 — data/](#ii2--data--processed-data)
  - [II.3 — docs/prompt_engineering.md](#ii3--docsprompt_engineeringmd--prompt-engineering-documentation)
  - [II.4 — models/](#ii4--models--saved-artifacts)
  - [II.5 — notebooks/eda.ipynb](#ii5--notebookstedaipynb--exploratory-data-analysis)
  - [II.6 — outputs/](#ii6--outputs--evaluation-plots)
  - [II.7 — src/data_processing.py](#ii7--srcdata_processingpy--preprocessing-pipeline)
  - [II.8 — src/train_model.py](#ii8--srctrain_modelpy--model-training--comparison)
  - [II.9 — src/evaluate_model.py](#ii9--srcevaluate_modelpy--deep-evaluation)
  - [II.10 — tests/](#ii10--tests--automated-tests)
  - [II.11 — Dockerfile](#ii11--dockerfile--containerization)
  - [II.12 — requirements.txt](#ii12--requirementstxt--dependencies)
- [III. Installation & Usage](#iii-installation--usage)
- [IV. Critical Questions](#iv-critical-questions)

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
**Team:** Data Healers — Amghar Douae, Azoud Hajar, Boutalmaouine Amina, Dyaz Hajar, Querchi Meryem  

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
    │   ├── data_processing.py
    │   ├── evaluate_model.py
    │   └── train_model.py
    │
    ├── tests/
    │   ├── __init__.py
    │   ├── test_app.py
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

**Model selection — why 3 models and not 4:**

The project brief recommended 4 models: Random Forest, XGBoost, LightGBM, and CatBoost. After analysis, **CatBoost was deliberately excluded** for the following reason: CatBoost is specifically designed for high-cardinality categorical features and large-scale datasets. Our dataset contains only ~2,111 rows with low-cardinality categorical features, a context where CatBoost provides no meaningful advantage over LightGBM while significantly increasing training time. The 3 selected models are fully representative and sufficient for this dataset size.

**Models trained and compared:**

| Model | Accuracy | F1-Score (Weighted) |
|---|---|---|
| Random Forest | ~0.94 | ~0.94 |
| XGBoost | ~0.95 | ~0.95 |
| **LightGBM ✅** | **~0.96** | **~0.96** |

**Why LightGBM was selected as the final model:**
- Highest accuracy and F1-score across all evaluation runs
- Faster training than XGBoost on this dataset size
- Native SHAP compatibility via `TreeExplainer`
- Less prone to overfitting with default parameters
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

5 test files executed automatically via GitHub Actions on every push and pull request.

| File | What it verifies |
|---|---|
| `test_app.py` | Streamlit app loads correctly, all required model files are present and accessible at startup |
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

- Python 3.11+

### 1. Clone the repository

```bash
git clone https://github.com/Douae142005/app-obesity-risk-prediction.git
cd app-obesity-risk-prediction/project
```
