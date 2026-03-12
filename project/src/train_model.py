"""
train_model.py
--------------
Entraîne plusieurs modèles de machine learning pour la prédiction de l'obésité.
Modèles : Random Forest, XGBoost, LightGBM
Sauvegarde les modèles et leurs performances.
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CHARGEMENT DES DONNÉES PRÉTRAITÉES
# ============================================

def load_processed_data():
    """Charge les données déjà prétraitées par data_processing.py"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    
    # Créer les dossiers s'ils n'existent pas
    os.makedirs(models_dir, exist_ok=True)
    
    # Charger les données
    train_df = pd.read_csv(os.path.join(data_dir, 'train_processed.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_processed.csv'))
    
    # Séparer features et target
    X_train = train_df.drop('NObeyesdad', axis=1)
    y_train = train_df['NObeyesdad']
    X_test = test_df.drop('NObeyesdad', axis=1)
    y_test = test_df['NObeyesdad']
    
    print(f"[INFO] Données chargées :")
    print(f"      Train : {X_train.shape[0]} échantillons, {X_train.shape[1]} features")
    print(f"      Test  : {X_test.shape[0]} échantillons, {X_test.shape[1]} features")
    
    return X_train, X_test, y_train, y_test


    return X_train, X_test, y_train, y_test
# ============================================
# 2. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES
# ============================================

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """
    Entraîne un modèle et calcule ses métriques de performance.
    """
    print(f"\n{'='*50}")
    print(f"🚀 Entraînement du modèle : {model_name}")
    print(f"{'='*50}")
    
    # Mesurer le temps d'entraînement
    start_time = time.time()
    
    # Entraînement
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"⏱️  Temps d'entraînement : {training_time:.2f} secondes")
    
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métriques
    metrics = {
        'model_name': model_name,
        'training_time': training_time,
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision_macro': precision_score(y_test, y_pred_test, average='macro'),
        'test_recall_macro': recall_score(y_test, y_pred_test, average='macro'),
        'test_f1_macro': f1_score(y_test, y_pred_test, average='macro')
    }
    
    # Affichage des résultats
    print(f"\n📊 PERFORMANCES DU MODÈLE {model_name}:")
    print(f"   Accuracy Train : {metrics['train_accuracy']:.4f} ({metrics['train_accuracy']*100:.2f}%)")
    print(f"   Accuracy Test  : {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
    print(f"   Precision (macro) : {metrics['test_precision_macro']:.4f}")
    print(f"   Recall (macro)    : {metrics['test_recall_macro']:.4f}")
    print(f"   F1-Score (macro)  : {metrics['test_f1_macro']:.4f}")
    
    return model, metrics, y_pred_test
# ============================================
# 3. CRÉATION DES MODÈLES
# ============================================

def create_models(random_state=42):
    """Crée les 3 modèles avec leurs hyperparamètres."""
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        ),
        
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='mlogloss',
            use_label_encoder=False
        ),
        
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1
        )
    }
    
    return models
# ============================================
# 4. VISUALISATION DES RÉSULTATS
# ============================================

def plot_confusion_matrix(y_true, y_pred, model_name, classes):
    """Affiche et sauvegarde la matrice de confusion."""
    
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
    plt.savefig(os.path.join(models_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
    plt.show()


def plot_metrics_comparison(metrics_list):
    """Compare les métriques des différents modèles."""
    
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics.set_index('model_name')
    
    # Sélectionner les colonnes à comparer
    metrics_to_plot = ['test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']
    
    plt.figure(figsize=(12, 6))
    df_metrics[metrics_to_plot].plot(kind='bar', rot=0)
    plt.title('Comparaison des Performances des Modèles')
    plt.xlabel('Modèle')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.ylim(0.8, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarde
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(project_root, 'models')
    plt.savefig(os.path.join(models_dir, 'model_comparison.png'))
    plt.show()
    
    return df_metrics
