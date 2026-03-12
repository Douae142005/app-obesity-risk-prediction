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
# ============================================
# 2. PRÉDICTIONS
# ============================================
 
def make_predictions(model, X_test):
    """Génère y_pred (classes) et y_proba (probabilités)."""
 
    y_pred = model.predict(X_test)
    # y_proba nécessaire pour ROC-AUC et l'affichage du % confiance dans Streamlit
    y_proba = model.predict_proba(X_test)
 
    print(f"✅ Prédictions générées — {len(y_pred)} patients")
    return y_pred, y_proba
 # ============================================
# 3. MÉTRIQUES COMPLÈTES
# ============================================

def compute_metrics(y_test, y_pred, y_proba, classes):
    """Calcule et affiche toutes les métriques d'évaluation."""

    print("\n" + "="*60)
    print("📊 MÉTRIQUES GLOBALES")
    print("="*60)

    accuracy  = accuracy_score(y_test, y_pred)
    # F1-Macro : poids égal par classe — adapté car classes équilibrées (~12-15%)
    f1_macro  = f1_score(y_test, y_pred, average='macro')
    # F1-Weighted : pondéré par taille de classe — cohérent avec train_model.py
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # ROC-AUC nécessite y_proba — mesure la capacité de discrimination globale
    try:
        y_test_bin = label_binarize(y_test, classes=classes)
        roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"   ⚠️ ROC-AUC non calculé : {e}")
        roc_auc = None

    print(f"   Accuracy   : {accuracy:.4f}")
    print(f"   F1-Macro   : {f1_macro:.4f}")
    print(f"   F1-Weighted: {f1_weighted:.4f}")
    if roc_auc:
        print(f"   ROC-AUC    : {roc_auc:.4f}")

    # Interprétation automatique — alerte si performances insuffisantes
    print("\n🩺 Interprétation clinique :")
    if roc_auc:
        if roc_auc >= 0.95:
            print("   🟢 Excellent — modèle fiable pour aide à la décision")
        elif roc_auc >= 0.85:
            print("   🟡 Correct — supervision médicale recommandée")
        else:
            print("   🔴 Insuffisant — ne pas utiliser en production")

    # Rapport détaillé par classe (precision/recall/F1 pour chacune des 7 classes)
    print("\n" + "="*60)
    print("📋 RAPPORT PAR CLASSE")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))

    return {
        'accuracy'   : accuracy,
        'f1_macro'   : f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc'    : roc_auc
    }
