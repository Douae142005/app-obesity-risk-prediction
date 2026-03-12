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

# ============================================
# 4. ENCODAGE DES VARIABLES CATÉGORIELLES
# ============================================

def encode_features(df):
    """Encode les variables catégorielles en valeurs numériques."""
    df = df.copy()
    label_encoders = {}

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"[INFO] Colonnes à encoder : {categorical_cols}")

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    print("[INFO] Encodage terminé ✅")
    return df, label_encoders

# ============================================
# 5. NORMALISATION DES COLONNES NUMÉRIQUES
# ============================================

def normalize_features(X_train, X_test):
    """
    Normalise les colonnes numériques avec StandardScaler.
    - Fit sur X_train uniquement (pour éviter le data leakage)
    - Transform sur X_train et X_test
    """
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    cols_to_scale = [col for col in numerical_cols if col in X_train.columns]

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    print(f"[INFO] Normalisation appliquée sur : {cols_to_scale} ✅")
    return X_train, X_test, scaler


# ============================================
# 6. SÉPARATION FEATURES / TARGET
# ============================================

def split_features_target(df, target_col='NObeyesdad'):
    """Sépare les features (X) de la variable cible (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"[INFO] Features : {X.shape[1]} colonnes | Target : {target_col}")
    return X, y


# ============================================
# 7. SPLIT TRAIN / TEST
# ============================================

def split_train_test(X, y, test_size=0.2, random_state=42):
    """Divise les données en ensembles d'entraînement et de test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ============================================
# 8. SAUVEGARDE DES DONNÉES ET ENCODERS
# ============================================

def save_processed_data(X_train, X_test, y_train, y_test, label_encoders, scaler):
    """
    Sauvegarde :
    - Le dataset traité en CSV dans data/
    - Les encoders et scaler avec joblib pour l'app Streamlit
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Sauvegarde CSV
    train_df = X_train.copy()
    train_df['NObeyesdad'] = y_train.values
    train_df.to_csv(os.path.join(data_dir, 'train_processed.csv'), index=False)

    test_df = X_test.copy()
    test_df['NObeyesdad'] = y_test.values
    test_df.to_csv(os.path.join(data_dir, 'test_processed.csv'), index=False)

    print(f"[INFO] Données sauvegardées dans {data_dir} ✅")

    # Sauvegarde encoders et scaler
    joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

    print(f"[INFO] Encoders et scaler sauvegardés dans {models_dir} ✅")


# ============================================
# 9. PIPELINE COMPLET
# ============================================

def preprocess_pipeline():
    """Pipeline complet de prétraitement des données."""
    df = load_data()
    df = handle_missing_values(df)
    df = optimize_memory(df)
    df, label_encoders = encode_features(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train, X_test, scaler = normalize_features(X_train, X_test)
    save_processed_data(X_train, X_test, y_train, y_test, label_encoders, scaler)

    return X_train, X_test, y_train, y_test, label_encoders, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoders, scaler = preprocess_pipeline()
    print("\n✅ Prétraitement terminé avec succès !")
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    