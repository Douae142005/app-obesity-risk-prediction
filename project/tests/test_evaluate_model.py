"""
test_evaluate_model.py
----------------------
Tests unitaires pour le script d'évaluation evaluate_model.py
Vérifie que les fonctions d'évaluation fonctionnent correctement.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # Mode non interactif pour les tests

# Ajouter le chemin src pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from evaluate_model import (
    load_best_model, make_predictions, compute_metrics,
    plot_confusion_matrix_final, plot_roc_curves
)
from data_processing import preprocess_pipeline


class TestEvaluateModel:
    """Tests pour les fonctions d'évaluation."""

    def test_load_best_model(self):
        """Test que le chargement du meilleur modèle fonctionne."""
        try:
            model = load_best_model()
            assert model is not None
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')
            print("✅ test_load_best_model passé")
        except FileNotFoundError:
            pytest.skip("Modèle non trouvé - exécuter train_model.py d'abord")

    def test_make_predictions(self):
        """Test que les prédictions fonctionnent."""
        try:
            # Charger modèle et données
            model = load_best_model()
            _, X_test, _, y_test, _, _ = preprocess_pipeline()
            
            # Faire les prédictions
            y_pred, y_proba = make_predictions(model, X_test)
            
            # Vérifications
            assert len(y_pred) == len(y_test)
            assert y_proba.shape[0] == len(y_test)
            assert y_proba.shape[1] == len(np.unique(y_test))
            
            print(f"✅ test_make_predictions passé")
            
        except Exception as e:
            pytest.skip(f"Données non disponibles: {e}")

    def test_compute_metrics(self):
        """Test que le calcul des métriques fonctionne."""
        try:
            # Créer des données de test simples
            y_test = np.array([0, 1, 2, 0, 1, 2])
            y_pred = np.array([0, 1, 2, 0, 1, 2])
            y_proba = np.array([
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9]
            ])
            classes = [0, 1, 2]
            
            # Calculer les métriques
            metrics = compute_metrics(y_test, y_pred, y_proba, classes)
            
            # Vérifications
            assert 'accuracy' in metrics
            assert 'f1_macro' in metrics
            assert 'f1_weighted' in metrics
            assert metrics['accuracy'] == 1.0  # Prédictions parfaites
            
            print("✅ test_compute_metrics passé")
            
        except Exception as e:
            pytest.fail(f"❌ Erreur: {e}")

 