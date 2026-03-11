"""
data_processing.py
------------------
Chargement, nettoyage, encodage, normalisation,
optimisation mémoire et sauvegarde
du dataset Obesity Risk Estimation.
"""

import pandas as pd # Tableaux de données
import numpy as np # Calculs numériques
import os # Gestion fichiers
import joblib  # Sauvegarde modèle
from ucimlrepo import fetch_ucirepo # Dataset UCI
from sklearn.preprocessing import LabelEncoder, StandardScaler # Encodage + Normalisation
from sklearn.model_selection import train_test_split # Split 80/20


# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================

def load_data():
    """Charge le dataset depuis UCI repository."""
    dataset = fetch_ucirepo(id=544)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    print(f"[INFO] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# ============================================
# 2. VÉRIFICATION DES VALEURS MANQUANTES
# ============================================

def handle_missing_values(df):
    """Vérifie et traite les valeurs manquantes."""
    missing = df.isnull().sum().sum()
    if missing == 0:
        print("[INFO] Aucune valeur manquante détectée ✅")
    else:
        print(f"[INFO] {missing} valeurs manquantes → remplissage par médiane/mode")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
    return df


# ============================================
# 3. OPTIMISATION MÉMOIRE
# ============================================

def optimize_memory(df):
    """
    Optimise l'utilisation mémoire du DataFrame
    en réduisant les types de données :
    - float64 → float32
    - int64   → int32
    """
    before = df.memory_usage(deep=True).sum() / 1024
    print(f"[INFO] Mémoire avant optimisation : {before:.2f} KB")

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    after = df.memory_usage(deep=True).sum() / 1024
    print(f"[INFO] Mémoire après optimisation  : {after:.2f} KB")
    print(f"[INFO] Réduction : {((before - after) / before * 100):.1f}% ✅")

    return df
