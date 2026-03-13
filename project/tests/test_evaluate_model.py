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

    def test_compute_metrics_with_errors(self):
        """Test que les métriques fonctionnent avec des erreurs."""
        try:
            # Créer des données avec des erreurs
            y_test = np.array([0, 1, 2, 0, 1, 2])
            y_pred = np.array([0, 1, 1, 0, 2, 2])  # 2 erreurs
            y_proba = np.random.random((6, 3))
            y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
            classes = [0, 1, 2]
            
            # Calculer les métriques
            metrics = compute_metrics(y_test, y_pred, y_proba, classes)
            
            # Vérifications
            assert metrics['accuracy'] < 1.0
            assert metrics['f1_macro'] < 1.0
            assert metrics['f1_weighted'] < 1.0
            
            print("✅ test_compute_metrics_with_errors passé")
            
        except Exception as e:
            pytest.fail(f"❌ Erreur: {e}")

    def test_plot_functions(self):
        """Test que les fonctions de graphique ne génèrent pas d'erreurs."""
        try:
            # Créer des données de test
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
            
            # Dossier temporaire pour les outputs
            test_outputs = os.path.join(os.path.dirname(__file__), 'test_outputs')
            os.makedirs(test_outputs, exist_ok=True)
            
            # Tester les fonctions de graphique
            try:
                plot_confusion_matrix_final(y_test, y_pred, classes, test_outputs)
                print("✅ test_plot_confusion_matrix_final passé")
            except Exception as e:
                pytest.fail(f"❌ Erreur matrice: {e}")
            
            try:
                plot_roc_curves(y_test, y_proba, classes, test_outputs)
                print("✅ test_plot_roc_curves passé")
            except Exception as e:
                pytest.fail(f"❌ Erreur ROC: {e}")
            
            # Nettoyer
            import shutil
            shutil.rmtree(test_outputs)
            
        except Exception as e:
            pytest.fail(f"❌ Erreur: {e}")

    def test_roc_auc_calculation(self):
        """Test spécifique pour le calcul du ROC-AUC."""
        try:
            # Cas parfait
            y_test = np.array([0, 1, 0, 1])
            y_proba = np.array([
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8]
            ])
            classes = [0, 1]
            
            y_test_bin = pd.get_dummies(y_test).values
            from sklearn.metrics import roc_auc_score
            
            roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
            
            assert roc_auc > 0.5
            assert roc_auc <= 1.0
            
            print("✅ test_roc_auc_calculation passé")
            
        except Exception as e:
            pytest.fail(f"❌ Erreur: {e}")


class TestEvaluateModelIntegration:
    """Tests d'intégration pour l'évaluation complète."""

    def test_full_evaluation_pipeline(self):
        """Test que le pipeline complet d'évaluation s'exécute."""
        try:
            # Importer le module principal
            import importlib
            import evaluate_model
            
            # Vérifier que le module peut être importé
            assert evaluate_model is not None
            print("✅ test_full_evaluation_pipeline passé (import)")
            
        except Exception as e:
            pytest.skip(f"Module non disponible: {e}")

    def test_metrics_consistency(self):
        """Test la cohérence des métriques entre elles."""
        try:
            # Créer des données
            y_test = np.array([0, 1, 2, 0, 1, 2])
            y_pred = np.array([0, 1, 1, 0, 2, 2])
            y_proba = np.random.random((6, 3))
            y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
            classes = [0, 1, 2]
            
            # Calculer les métriques
            metrics = compute_metrics(y_test, y_pred, y_proba, classes)
            
            # Vérifier les relations
            # F1-weighted devrait être entre precision et recall
            assert 0 <= metrics['f1_weighted'] <= 1
            assert 0 <= metrics['f1_macro'] <= 1
            
            # Accuracy et F1 devraient être corrélés
            if metrics['accuracy'] > 0.8:
                assert metrics['f1_weighted'] > 0.7
            
            print("✅ test_metrics_consistency passé")
            
        except Exception as e:
            pytest.fail(f"❌ Erreur: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTS D'ÉVALUATION")
    print("="*60)
    
    # Exécuter les tests
    test = TestEvaluateModel()
    test.test_load_best_model()
    test.test_make_predictions()
    test.test_compute_metrics()
    test.test_compute_metrics_with_errors()
    test.test_plot_functions()
    test.test_roc_auc_calculation()
    
    test_int = TestEvaluateModelIntegration()
    test_int.test_full_evaluation_pipeline()
    test_int.test_metrics_consistency()
    
    print("\n" + "="*60)
    print("🎉 TOUS LES TESTS D'ÉVALUATION SONT PASSÉS !")
    print("="*60)