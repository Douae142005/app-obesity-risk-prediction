"""
test_model.py
-------------
Tests unitaires pour l'entraînement et l'évaluation des modèles.
"""

import os
import sys
import joblib
import pandas as pd

# Ajouter le chemin src pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_model import create_models

class TestModelTraining:
    """Tests pour l'entraînement des modèles."""

    def test_create_models(self):
        """Test que les modèles sont correctement créés."""
        models = create_models()
        
        assert len(models) == 3
        assert 'Random Forest' in models
        assert 'XGBoost' in models
        assert 'LightGBM' in models
        print("✅ test_create_models passé")

    def test_model_files_exist(self):
        """Test que les fichiers modèles existent après entraînement."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(project_root, 'models')
        
        model_path = os.path.join(models_dir, 'best_model.pkl')
        assert os.path.exists(model_path), "❌ best_model.pkl n'existe pas"
        
        encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
        assert os.path.exists(encoders_path), "❌ label_encoders.pkl n'existe pas"
        
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        assert os.path.exists(scaler_path), "❌ scaler.pkl n'existe pas"
        
        print("✅ test_model_files_exist passé")

    def test_model_prediction(self):
        """Test que le modèle peut faire des prédictions."""
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            model_path = os.path.join(project_root, 'models', 'best_model.pkl')
            model = joblib.load(model_path)
            
            X_test = pd.DataFrame({
                'Age': [30],
                'Gender': [1],
                'Height': [1.75],
                'Weight': [70],
                'family_history_with_overweight': [1],
                'FAVC': [0],
                'FCVC': [2],
                'NCP': [3],
                'CAEC': [2],
                'SMOKE': [0],
                'CH2O': [2],
                'SCC': [0],
                'FAF': [1],
                'TUE': [1],
                'CALC': [1],
                'MTRANS': [3]
            })
            
            prediction = model.predict(X_test)
            
            assert len(prediction) == 1
            assert prediction[0] in range(7)
            print(f"✅ test_model_prediction passé (prédiction: {prediction[0]})")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            raise

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧪 TESTS DES MODÈLES")
    print("="*50)
    
    test = TestModelTraining()
  test.test_model_prediction()
    
    print("\n" + "="*50)
    print("🎉 TOUS LES TESTS MODÈLES SONT PASSÉS !")
    print("="*50) 



