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