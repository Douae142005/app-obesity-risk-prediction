"""
evaluate_model.py
-----------------
Évaluation approfondie du meilleur modèle sauvegardé par train_model.py.
Génère :
- Métriques complètes (Accuracy, F1-Macro, F1-Weighted, ROC-AUC)
- Matrice de confusion finale
- Courbes ROC (une par classe)
- Interprétation automatique des résultats

Prérequis : avoir exécuté train_model.py → models/best_model.pkl
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Même import que train_model.py — cohérence du pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import preprocess_pipeline

print("="*60)
print("🔍 DÉMARRAGE DE L'ÉVALUATION")
print("="*60)


# ============================================
# 1. CHARGEMENT DU MODÈLE SAUVEGARDÉ
# ============================================

def load_best_model():
    """Charge best_model.pkl sauvegardé par train_model.py."""

    # Chemin identique à save_best_model() dans train_model.py
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, 'models', 'best_model.pkl')

    if not os.path.exists(model_path):
        # train_model.py doit être exécuté en premier
        raise FileNotFoundError(
            f"❌ Modèle introuvable : {model_path}\n"
            "   → Lance d'abord : python src/train_model.py"
        )

    model = joblib.load(model_path)
    print(f"✅ Modèle chargé : {model_path}")
    print(f"   Type : {type(model).__name__}")
    return model
