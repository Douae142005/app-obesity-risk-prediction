
"""
train_model.py
--------------
Entraînement et comparaison de 3 modèles ML :
- Random Forest
- XGBoost
- LightGBM

Sauvegarde le meilleur modèle dans models/best_model.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report,
                             confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Import du pipeline de prétraitement
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import preprocess_pipeline

print("="*60)
print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT")
print("="*60)


# ============================================
# 1. FONCTION POUR LA MATRICE DE CONFUSION
# ============================================

def plot_confusion_matrix(y_true, y_pred, model_name, classes):
    """Affiche et sauvegarde la matrice de confusion."""
    
    try:
        print(f"   📊 Création matrice pour {model_name}...")
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'Matrice de Confusion - {model_name}')
        plt.xlabel('Prédit')
        plt.ylabel('Réel')
        plt.tight_layout()
        
        # Sauvegarde
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(models_dir, filename))
        plt.close()
        print(f"   ✅ Matrice sauvegardée: {filename}")
    except Exception as e:
        print(f"   ⚠️ Erreur matrice: {e}")


# ============================================
# 2. ÉVALUATION D'UN MODÈLE
# ============================================

def evaluate_model(model, X_test, y_test, model_name, classes):
    """Calcule les métriques et affiche la matrice de confusion."""
    
    print(f"\n📊 Évaluation de {model_name}...")
    
    try:
        y_pred = model.predict(X_test)
        print(f"   ✅ Prédictions faites")

        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")

        # Rapport de classification
        print(f"\n{classification_report(y_test, y_pred)}")
        
        # Matrice de confusion
        plot_confusion_matrix(y_test, y_pred, model_name, classes)

        return {
            'model_name': model_name,
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1
        }
    except Exception as e:
        print(f"❌ Erreur évaluation {model_name}: {e}")
        return None


