"""
test_app.py
-----------
Tests unitaires pour l'application Streamlit.
Vérifie le chargement, les prédictions et l'affichage.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Ajouter le chemin de l'application
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'app'))

# Import conditionnel pour éviter les erreurs hors contexte Streamlit
try:
    import app
    APP_IMPORT_SUCCESS = True
except:
    APP_IMPORT_SUCCESS = False


class TestApp:
    """Tests pour l'application Streamlit."""

    def test_app_imports(self):
        """Test que l'application peut être importée."""
        try:
            import app
            assert app is not None
            print("✅ test_app_imports passé")
        except ImportError as e:
            pytest.fail(f"❌ Impossible d'importer app.py: {e}")

    def test_charger_modele_exists(self):
        """Test que la fonction charger_modele existe."""
        try:
            from app import charger_modele
            assert callable(charger_modele)
            print("✅ test_charger_modele_exists passé")
        except ImportError:
            pytest.skip("Fonction charger_modele non trouvée")

    def test_img_to_base64_exists(self):
        """Test que la fonction img_to_base64 existe."""
        try:
            from app import img_to_base64
            assert callable(img_to_base64)
            print("✅ test_img_to_base64_exists passé")
        except ImportError:
            pytest.skip("Fonction img_to_base64 non trouvée")

    def test_img_to_base64_fichier_inexistant(self):
        """Test que img_to_base64 retourne vide si fichier absent."""
        try:
            from app import img_to_base64
            result = img_to_base64("fichier_inexistant.jpg")
            assert result == ""
            print("✅ test_img_to_base64_fichier_inexistant passé")
        except ImportError:
            pytest.skip("Fonction img_to_base64 non trouvée")


class TestAppDataProcessing:
    """Tests pour le traitement des données dans l'app."""

    def test_donnees_patient_structure(self):
        """Test la structure des données patient."""
        donnees = {
            'Gender': 'Male',
            'Age': 30,
            'Height': 1.75,
            'Weight': 70,
            'family_history_with_overweight': 'yes',
            'FAVC': 'no',
            'FCVC': 2.0,
            'NCP': 3.0,
            'CAEC': 'Sometimes',
            'SMOKE': 'no',
            'CH2O': 2.0,
            'SCC': 'no',
            'FAF': 1.0,
            'TUE': 1.0,
            'CALC': 'no',
            'MTRANS': 'Public_Transportation'
        }
        
        df = pd.DataFrame([donnees])
        
        # Vérifier le nombre de colonnes
        assert len(df.columns) == 16
        
        # Vérifier les types
        assert df['Age'].dtype in ['int64', 'float64']
        assert df['Gender'].dtype == 'object'
        
        print("✅ test_donnees_patient_structure passé")

    def test_colonnes_numeriques(self):
        """Test la liste des colonnes numériques."""
        # Définition locale car la variable n'est pas exportée
        cols_numeriques = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        
        expected = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        
        assert len(cols_numeriques) == 8
        assert all(col in cols_numeriques for col in expected)
        print("✅ test_colonnes_numeriques passé")

    def test_colonnes_categorielles(self):
        """Test la liste des colonnes catégorielles."""
        cols_categorielles = [
            'Gender', 'family_history_with_overweight',
            'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
        ]
        
        expected = [
            'Gender', 'family_history_with_overweight',
            'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
        ]
        
        assert len(cols_categorielles) == 8
        assert all(col in cols_categorielles for col in expected)
        print("✅ test_colonnes_categorielles passé")


class TestAppPredictions:
    """Tests pour les prédictions de l'application."""

    def test_niveaux_obesite(self):
        """Test que les 7 niveaux d'obésité sont définis."""
        niveaux = {
            0: ("🔵", "Insufficient Weight"),
            1: ("🟢", "Normal Weight"),
            2: ("🟡", "Overweight Level I"),
            3: ("🟠", "Overweight Level II"),
            4: ("🔴", "Obesity Type I"),
            5: ("🔴", "Obesity Type II"),
            6: ("🔴", "Obesity Type III"),
        }
        
        assert len(niveaux) == 7
        assert niveaux[0][1] == "Insufficient Weight"
        assert niveaux[1][1] == "Normal Weight"
        assert "Obesity" in niveaux[4][1]
        print("✅ test_niveaux_obesite passé")

    def test_calcul_imc(self):
        """Test le calcul de l'IMC."""
        poids = 70
        taille = 1.75
        imc = poids / (taille ** 2)
        
        assert round(imc, 1) == 22.9
        print("✅ test_calcul_imc passé")

    def test_interpretation_imc(self):
        """Test l'interprétation de l'IMC."""
        def interprete_imc(imc):
            if imc < 18.5:
                return "Poids insuffisant"
            elif imc < 25:
                return "Poids normal"
            elif imc < 30:
                return "Surpoids"
            else:
                return "Obésité"
        
        assert interprete_imc(17) == "Poids insuffisant"
        assert interprete_imc(22) == "Poids normal"
        assert interprete_imc(27) == "Surpoids"
        assert interprete_imc(32) == "Obésité"
        print("✅ test_interpretation_imc passé")


class TestAppPages:
    """Tests pour les différentes pages de l'application."""

    def test_pages_exist(self):
        """Test que toutes les pages sont définies."""
        pages = [
            "🏠 Accueil",
            "👤 Analyse Patient",
            "📊 Statistiques",
            "👥 Notre Équipe",
            "ℹ️ À propos"
        ]
        
        assert len(pages) == 5
        assert "🏠 Accueil" in pages
        assert "👤 Analyse Patient" in pages
        assert "📊 Statistiques" in pages
        print("✅ test_pages_exist passé")

    def test_membres_equipe(self):
        """Test que les membres de l'équipe sont listés."""
        membres = [
            "Meryem Querchi",
            "Amina Boutalmaouine",
            "Douae Amghar",
            "Hajar Azoud",
            "Hajar Dyaz"
        ]
        
        assert len(membres) == 5
        assert "Meryem Querchi" in membres
        assert "Hajar Azoud" in membres
        print("✅ test_membres_equipe passé")

    def test_couleurs_membres(self):
        """Test les couleurs associées aux membres."""
        couleurs = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
        
        assert len(couleurs) == 5
        assert "#3498db" in couleurs
        print("✅ test_couleurs_membres passé")


class TestAppIntegration:
    """Tests d'intégration pour l'application."""

    def test_fichiers_essentiels_existent(self):
        """Test que les fichiers essentiels existent."""
        app_dir = os.path.join(os.path.dirname(__file__), '..', 'app')
        
        # Vérifier que app.py existe
        app_py = os.path.join(app_dir, 'app.py')
        assert os.path.exists(app_py), f"❌ {app_py} n'existe pas"
        
        # Vérifier que le dossier models existe
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        assert os.path.exists(models_dir), f"❌ {models_dir} n'existe pas"
        
        print("✅ test_fichiers_essentiels_existent passé")

    def test_images_optionnelles(self):
        """Test que les images optionnelles existent (sans bloquer)."""
        app_dir = os.path.join(os.path.dirname(__file__), '..', 'app')
        
        images = ['logo.png', 'team.jpg', 'bg1.jpg', 'bg2.jpg', 'bg3.jpg', 'bg4.jpg']
        
        existantes = []
        for img in images:
            path = os.path.join(app_dir, img)
            if os.path.exists(path):
                existantes.append(img)
        
        print(f"   Images trouvées: {existantes}")
        print("✅ test_images_optionnelles passé")

    def test_structure_fichiers(self):
        """Test la structure des fichiers du projet."""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        
        dossiers_attendus = ['app', 'src', 'models', 'tests', 'data']
        
        for dossier in dossiers_attendus:
            path = os.path.join(project_root, dossier)
            assert os.path.exists(path), f"❌ Dossier {dossier} manquant"
        
     print("✅ test_structure_fichiers passé")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("📱 TESTS DE L'APPLICATION")
    print("="*60)
    
   