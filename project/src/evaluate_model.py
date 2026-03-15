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
# ============================================
# 4. MATRICE DE CONFUSION FINALE
# ============================================

def plot_confusion_matrix_final(y_test, y_pred, classes, outputs_dir):
    """Matrice de confusion du meilleur modèle sur X_test complet."""

    try:
        print("\n📊 Génération matrice de confusion finale...")
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)

        # Normalisation en % pour lisibilité (annot affiche les valeurs brutes aussi)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)

        plt.title('Matrice de Confusion — Meilleur Modèle (Test Set)')
        plt.xlabel('Prédit')
        plt.ylabel('Réel')
        plt.tight_layout()

        path = os.path.join(outputs_dir, 'confusion_matrix_final.png')
        plt.savefig(path)
        plt.close()
        print(f"   ✅ Sauvegardée : {path}")

    except Exception as e:
        print(f"   ⚠️ Erreur matrice : {e}")

# ============================================
# 5. COURBES ROC
# ============================================

def plot_roc_curves(y_test, y_proba, classes, outputs_dir):
    """Trace une courbe ROC par classe (One-vs-Rest) — nécessite y_proba."""

    try:
        print("\n📈 Génération courbes ROC...")
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes  = len(classes)

        plt.figure(figsize=(10, 7))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc_i   = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, label=f"{classes[i]} (AUC={roc_auc_i:.2f})")

        # Diagonale = modèle aléatoire (AUC=0.5) — référence visuelle
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Aléatoire (AUC=0.50)')
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbes ROC — Une courbe par classe (One-vs-Rest)')
        plt.legend(loc='lower right', fontsize=8)
        plt.tight_layout()

        path = os.path.join(outputs_dir, 'roc_curves.png')
        plt.savefig(path)
        plt.close()
        print(f"   ✅ Sauvegardée : {path}")

    except Exception as e:
        print(f"   ⚠️ Erreur ROC : {e}")
# ============================================
# 6. EXÉCUTION PRINCIPALE
# ============================================

if __name__ == "__main__":

    try:
        # --- Données : même pipeline que train_model.py (random_state=42 garanti) ---
        print("\n📥 Chargement des données...")
        X_train, X_test, y_train, y_test, label_encoders, scaler = preprocess_pipeline()
        print(f"✅ Données prêtes — Test : {X_test.shape}")

        classes = sorted(y_test.unique())

        # --- Dossier outputs pour les graphiques ---
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        outputs_dir  = os.path.join(project_root, 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)

        # --- Pipeline d'évaluation ---
        model            = load_best_model()
        y_pred, y_proba  = make_predictions(model, X_test)
        metrics          = compute_metrics(y_test, y_pred, y_proba, classes)

        plot_confusion_matrix_final(y_test, y_pred, classes, outputs_dir)
        plot_roc_curves(y_test, y_proba, classes, outputs_dir)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"\n❌ Erreur générale : {e}")

    print("\n" + "="*60)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*60)

