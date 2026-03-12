# ============================================================
# tests/test_data_processing.py
# ============================================================
# RÔLE : Vérifier que chaque fonction de data_processing.py
# fonctionne correctement de façon isolée.
#
# RELATION AVEC CI/CD :
# Ces tests sont écrits UNE FOIS ici.
# Le CI (.github/workflows/ci.yml) les exécute
# AUTOMATIQUEMENT à chaque push sur GitHub.
#
# Comment lancer manuellement :
#   pytest tests/test_data_processing.py -v
# ============================================================

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajouter src/ au path pour pouvoir importer data_processing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_processing import preprocess_pipeline


# ============================================
# FIXTURE — Données de test partagées
# ============================================

@pytest.fixture
def sample_data():
    """
    Crée un mini DataFrame qui simule le dataset UCI.
    Utilisé par tous les tests ci-dessous.
    On ne télécharge pas le vrai dataset dans les tests
    → plus rapide, pas besoin d'internet dans le CI.
    """
    return pd.DataFrame({
        'Gender'                        : ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Age'                           : [21.0, 25.0, 30.0, 22.0, 28.0],
        'Height'                        : [1.75, 1.62, 1.80, 1.58, 1.70],
        'Weight'                        : [77.0, 65.0, 90.0, 55.0, 85.0],
        'family_history_with_overweight': ['yes', 'no', 'yes', 'no', 'yes'],
        'FAVC'                          : ['yes', 'no', 'yes', 'no', 'yes'],
        'FCVC'                          : [2.0, 3.0, 2.0, 3.0, 2.0],
        'NCP'                           : [3.0, 2.0, 3.0, 1.0, 3.0],
        'CAEC'                          : ['Sometimes', 'Frequently', 'no', 'Always', 'Sometimes'],
        'SMOKE'                         : ['no', 'yes', 'no', 'no', 'no'],
        'CH2O'                          : [2.0, 3.0, 2.0, 1.0, 2.0],
        'SCC'                           : ['no', 'yes', 'no', 'no', 'no'],
        'FAF'                           : [0.0, 3.0, 2.0, 1.0, 0.0],
        'TUE'                           : [1.0, 0.0, 1.0, 0.0, 1.0],
        'CALC'                          : ['no', 'Sometimes', 'Frequently', 'Always', 'no'],
        'MTRANS'                        : ['Public_Transportation', 'Walking',
                                          'Automobile', 'Bike', 'Public_Transportation'],
        'NObeyesdad'                    : ['Normal_Weight', 'Overweight_Level_I',
                                          'Obesity_Type_I', 'Insufficient_Weight',
                                          'Overweight_Level_II']
    })


# ============================================
# TEST 1 — Vérification du shape du DataFrame
# ============================================

def test_dataframe_shape(sample_data):
    """
    Vérifie que le DataFrame a bien 17 colonnes.
    Si quelqu'un supprime une colonne dans data_processing.py
    → ce test échoue → le CI bloque le push.
    """
    assert sample_data.shape[1] == 17, \
        f"Attendu 17 colonnes, obtenu {sample_data.shape[1]}"


# ============================================
# TEST 2 — Pas de valeurs manquantes
# ============================================

def test_no_missing_values(sample_data):
    """
    Vérifie qu'il n'y a aucun NaN dans le DataFrame.
    Relation avec data_processing.py :
    → handle_missing_values() doit garantir 0 NaN en sortie.
    """
    assert sample_data.isnull().sum().sum() == 0, \
        "Des valeurs manquantes ont été détectées"

# ============================================
# TEST 3 — Colonnes attendues présentes
# ============================================

def test_expected_columns(sample_data):
    """
    Vérifie que toutes les colonnes nécessaires sont présentes.
    Si une colonne est renommée par erreur → test échoue.
    """
    expected_cols = [
        'Gender', 'Age', 'Height', 'Weight',
        'family_history_with_overweight', 'FAVC', 'FCVC',
        'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF',
        'TUE', 'CALC', 'MTRANS', 'NObeyesdad'
    ]
    for col in expected_cols:
        assert col in sample_data.columns, f"Colonne manquante : {col}"
# ============================================
# TEST 4 — Types des colonnes numériques
# ============================================

def test_numeric_columns_type(sample_data):
    """
    Vérifie que les colonnes numériques sont bien en float.
    Relation avec optimize_memory() dans data_processing.py :
    → après optimisation, elles doivent rester numériques.
    """
    numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(sample_data[col]), \
            f"Colonne {col} devrait être numérique"

# ============================================
# TEST 5 — Valeurs de la colonne cible valides
# ============================================

def test_target_values(sample_data):
    """
    Vérifie que NObeyesdad ne contient que des classes connues.
    Si une classe inconnue apparaît → l'encodeur plantera en production.
    """
    valid_classes = {
        'Insufficient_Weight', 'Normal_Weight',
        'Overweight_Level_I', 'Overweight_Level_II',
        'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    }
    actual_classes = set(sample_data['NObeyesdad'].unique())
    unknown = actual_classes - valid_classes
    assert len(unknown) == 0, f"Classes inconnues détectées : {unknown}"
# ============================================
# TEST 6 — Valeurs binaires cohérentes
# ============================================

def test_binary_columns_values(sample_data):
    """
    Vérifie que FAVC, SMOKE, SCC ne contiennent que 'yes' ou 'no'.
    Relation avec encode_categorical() :
    → le map {'yes':1, 'no':0} plante si une autre valeur existe.
    """
    binary_cols = ['FAVC', 'SMOKE', 'SCC']
    valid_values = {'yes', 'no'}
    for col in binary_cols:
        actual = set(sample_data[col].unique())
        invalid = actual - valid_values
        assert len(invalid) == 0, \
            f"Colonne {col} contient des valeurs invalides : {invalid}"
# ============================================
# TEST 7 — Valeurs ordinales cohérentes
# ============================================

def test_ordinal_columns_values(sample_data):
    """
    Vérifie que CAEC et CALC ne contiennent que les 4 niveaux attendus.
    Relation avec encode_categorical() :
    → OrdinalEncoder échoue si un niveau inconnu est présent.
    """
    valid_ordinal = {'no', 'Sometimes', 'Frequently', 'Always'}
    for col in ['CAEC', 'CALC']:
        actual = set(sample_data[col].unique())
        invalid = actual - valid_ordinal
        assert len(invalid) == 0, \
            f"Colonne {col} contient des valeurs invalides : {invalid}"
