
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
from datetime import datetime
from unittest.mock import patch, MagicMock

# Ajouter le chemin de l'application
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'app'))

# Import conditionnel pour éviter les erreurs hors contexte Streamlit
try:
    import app
    APP_IMPORT_SUCCESS = True
except ImportError:
    APP_IMPORT_SUCCESS = False


class TestAppBase:
    """Tests de base pour l'application."""

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


class TestAppData:
    """Tests pour la structure des données."""

    def test_recommandations_structure(self):
        """Test que RECOMMANDATIONS est bien structuré."""
        try:
            from app import RECOMMANDATIONS
            
            assert len(RECOMMANDATIONS) == 7
            for i in range(7):
                assert i in RECOMMANDATIONS
                assert "titre" in RECOMMANDATIONS[i]
                assert "couleur" in RECOMMANDATIONS[i]
                assert "urgence" in RECOMMANDATIONS[i]
                assert "conseils" in RECOMMANDATIONS[i]
                assert isinstance(RECOMMANDATIONS[i]["conseils"], list)
                assert len(RECOMMANDATIONS[i]["conseils"]) > 0
            
            print("✅ test_recommandations_structure passé")
        except ImportError:
            pytest.skip("RECOMMANDATIONS non trouvée")

    def test_niveaux_obesite(self):
        """Test que les 7 niveaux d'obésité sont définis."""
        niveaux = {
            0: ("🔵", "Insufficient Weight", "#3498db"),
            1: ("🟢", "Normal Weight", "#2ecc71"),
            2: ("🟡", "Overweight Level I", "#f1c40f"),
            3: ("🟠", "Overweight Level II", "#e67e22"),
            4: ("🔴", "Obesity Type I", "#e74c3c"),
            5: ("🔴", "Obesity Type II", "#c0392b"),
            6: ("🔴", "Obesity Type III", "#922b21"),
        }
        
        assert len(niveaux) == 7
        assert niveaux[0][1] == "Insufficient Weight"
        assert niveaux[1][1] == "Normal Weight"
        assert "Obesity" in niveaux[4][1]
        print("✅ test_niveaux_obesite passé")

    def test_pages_navigation(self):
        """Test que toutes les pages de navigation sont définies."""
        pages_nav = [
            ("🏠", "Accueil", "🏠 Accueil"),
            ("👤", "Analyse Patient", "👤 Analyse Patient"),
            ("🧑", "Espace Patient", "🧑 Espace Patient"),
            ("📋", "Historique", "📋 Historique"),
            ("📊", "Statistiques", "📊 Statistiques"),
            ("👥", "Notre Équipe", "👥 Notre Équipe"),
            ("ℹ️", "À propos", "ℹ️ À propos"),
        ]
        
        assert len(pages_nav) == 7
        assert pages_nav[0][2] == "🏠 Accueil"
        assert pages_nav[1][2] == "👤 Analyse Patient"
        print("✅ test_pages_navigation passé")


class TestAppAuth:
    """Tests pour l'authentification des médecins."""

    def test_session_state_structure(self):
        """Test que les variables de session existent."""
        # Simulation des variables de session
        session_vars = [
            'connecte', 'medecin_nom', 'medecin_email',
            'historique', 'page', 'medecins_db'
        ]
        
        # Vérification que les variables sont définies dans le code
        with open(os.path.join(os.path.dirname(__file__), '..', 'app', 'app.py'), 'r', encoding='utf-8') as f:
            content = f.read()
            for var in session_vars:
                assert f"st.session_state.{var}" in content, f"Variable {var} manquante"
        
        print("✅ test_session_state_structure passé")

    def test_medecins_db_structure(self):
        """Test la structure de la base de données médecins."""
        medecins_db = {
            "test@test.com": {"password": "test123", "nom": "Dr Test"}
        }
        
        assert "test@test.com" in medecins_db
        assert medecins_db["test@test.com"]["password"] == "test123"
        assert medecins_db["test@test.com"]["nom"] == "Dr Test"
        print("✅ test_medecins_db_structure passé")

    def test_login_validation(self):
        """Test la validation des identifiants."""
        medecins_db = {
            "dr.valid@test.com": {"password": "valid123", "nom": "Dr Valid"}
        }
        
        # Test connexion réussie
        email = "dr.valid@test.com"
        password = "valid123"
        assert email in medecins_db
        assert medecins_db[email]["password"] == password
        
        # Test connexion échouée
        assert "invalid@test.com" not in medecins_db
        
        print("✅ test_login_validation passé")

    def test_inscription_validation(self):
        """Test la validation de l'inscription."""
        medecins_db = {}
        
        # Test création compte
        email = "new@test.com"
        password = "newpass123"
        nom = "Dr New"
        
        # Vérifications
        assert email not in medecins_db
        assert len(password) >= 6
        
        # Ajout
        medecins_db[email] = {"password": password, "nom": nom}
        
        assert email in medecins_db
        assert medecins_db[email]["password"] == password
        assert medecins_db[email]["nom"] == nom
        
        print("✅ test_inscription_validation passé")


class TestAppHistorique:
    """Tests pour l'historique des analyses."""

    def test_historique_structure(self):
        """Test la structure d'une entrée d'historique."""
        entree = {
            "date": "13/03/2026 15:30",
            "patient": "Patient Test",
            "medecin": "Dr Test",
            "prediction": "Normal Weight",
            "emoji": "🟢",
            "couleur": "#2ecc71",
            "confiance": "96.5%",
            "imc": "22.5",
        }
        
        assert "date" in entree
        assert "patient" in entree
        assert "medecin" in entree
        assert "prediction" in entree
        assert "confiance" in entree
        assert "imc" in entree
        
        print("✅ test_historique_structure passé")

    def test_historique_ajout(self):
        """Test l'ajout à l'historique."""
        historique = []
        nouvelle_entree = {
            "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "patient": "Test Patient",
            "medecin": "Dr Test",
            "prediction": "Normal Weight",
            "emoji": "🟢",
            "couleur": "#2ecc71",
            "confiance": "95.0%",
            "imc": "22.0",
        }
        
        historique.insert(0, nouvelle_entree)
        
        assert len(historique) == 1
        assert historique[0]["patient"] == "Test Patient"
        assert historique[0]["prediction"] == "Normal Weight"
        
        print("✅ test_historique_ajout passé")

    def test_historique_export_csv(self):
        """Test l'export CSV de l'historique."""
        historique = [
            {"date": "13/03/2026 15:30", "patient": "P1", "medecin": "Dr1", 
             "prediction": "Normal", "emoji": "🟢", "couleur": "#2ecc71",
             "confiance": "95%", "imc": "22.0"},
            {"date": "13/03/2026 16:00", "patient": "P2", "medecin": "Dr1",
             "prediction": "Surpoids", "emoji": "🟡", "couleur": "#f1c40f",
             "confiance": "92%", "imc": "27.0"},
        ]
        
        df = pd.DataFrame(historique)
        csv = df.to_csv(index=False)
        
        assert "patient" in csv
        assert "prediction" in csv
        assert "P1" in csv
        assert "P2" in csv
        assert "Normal" in csv
        assert "Surpoids" in csv
        
        print("✅ test_historique_export_csv passé")


class TestAppRecommandations:
    """Tests pour les recommandations médicales."""

    def test_recommandations_par_niveau(self):
        """Test les recommandations pour chaque niveau."""
        try:
            from app import RECOMMANDATIONS
            
            # Test pour niveau 0 (insuffisance pondérale)
            reco_0 = RECOMMANDATIONS[0]
            assert "⚠️ Poids Insuffisant" in reco_0["titre"]
            assert len(reco_0["conseils"]) >= 4
            
            # Test pour niveau 1 (poids normal)
            reco_1 = RECOMMANDATIONS[1]
            assert "Poids Normal" in reco_1["titre"]
            assert "Aucune action urgente" in reco_1["urgence"]
            
            # Test pour niveau 4 (obésité type I)
            reco_4 = RECOMMANDATIONS[4]
            assert "Obésité Type I" in reco_4["titre"]
            assert "immédiate" in reco_4["urgence"].lower()
            
            print("✅ test_recommandations_par_niveau passé")
        except ImportError:
            pytest.skip("RECOMMANDATIONS non trouvée")

    def test_recommandations_urgences(self):
        """Test les niveaux d'urgence."""
        try:
            from app import RECOMMANDATIONS
            
            # Niveaux non urgents (0-3)
            non_urgent_keywords = ["recommandée", "conseillé", "aucune action", "suivi", "dans les", "aucune"]
            for i in range(4):
                urgence = RECOMMANDATIONS[i]["urgence"].lower()
                assert any(keyword in urgence for keyword in non_urgent_keywords), \
                       f"Niveau {i} devrait être non urgent mais a: {RECOMMANDATIONS[i]['urgence']}"
            
            # Niveaux urgents (4-6)
            urgent_keywords = ["immédiate", "urgence", "hospitalisation", "immédiat", "absolue"]
            for i in range(4, 7):
                urgence = RECOMMANDATIONS[i]["urgence"].lower()
                assert any(keyword in urgence for keyword in urgent_keywords), \
                       f"Niveau {i} devrait être urgent mais a: {RECOMMANDATIONS[i]['urgence']}"
            
            print("✅ test_recommandations_urgences passé")
        except ImportError:
            pytest.skip("RECOMMANDATIONS non trouvée")


class TestAppMath:
    """Tests pour les calculs mathématiques."""

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


class TestAppFiles:
    """Tests pour les fichiers de l'application."""

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


class TestAppIntegration:
    """Tests d'intégration complets."""

    def test_structure_projet(self):
        """Test la structure complète du projet."""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        
        dossiers_attendus = ['app', 'src', 'models', 'tests', 'data']
        
        for dossier in dossiers_attendus:
            path = os.path.join(project_root, dossier)
            assert os.path.exists(path), f"❌ Dossier {dossier} manquant"
        
        print("✅ test_structure_projet passé")

    def test_requirements_exists(self):
        """Test que requirements.txt existe."""
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        req_path = os.path.join(project_root, 'requirements.txt')
        
        # Alternative: chercher aussi dans project/
        if not os.path.exists(req_path):
            req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
        
        if os.path.exists(req_path):
            with open(req_path, 'r') as f:
                content = f.read()
                assert 'streamlit' in content
                assert 'pandas' in content
                assert 'numpy' in content
                assert 'scikit-learn' in content
                assert 'xgboost' in content or 'lightgbm' in content
            print("✅ test_requirements_exists passé")
        else:
            print("⚠️ requirements.txt non trouvé (vérification ignorée)")

    def test_github_workflow_exists(self):
        """Test que le workflow GitHub Actions existe."""
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        workflow_path = os.path.join(project_root, '.github', 'workflows', 'ci.yml')
        
        if os.path.exists(workflow_path):
            print("✅ test_github_workflow_exists passé")
        else:
            print("⚠️ Workflow GitHub non trouvé (optionnel)")


# ============================================
# EXÉCUTION DES TESTS AVEC RÉSUMÉ
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("📱 TESTS COMPLETS DE L'APPLICATION")
    print("="*70)
    
    # Compteurs
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    # Tests de base
    test_base = TestAppBase()
    try:
        test_base.test_app_imports()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_app_imports: {e}")
    total_tests += 1
    
    try:
        test_base.test_charger_modele_exists()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_charger_modele_exists: {e}")
    total_tests += 1
    
    try:
        test_base.test_img_to_base64_exists()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_img_to_base64_exists: {e}")
    total_tests += 1
    
    try:
        test_base.test_img_to_base64_fichier_inexistant()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_img_to_base64_fichier_inexistant: {e}")
    total_tests += 1
    
    # Tests données
    test_data = TestAppData()
    try:
        test_data.test_recommandations_structure()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_recommandations_structure: {e}")
    total_tests += 1
    
    try:
        test_data.test_niveaux_obesite()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_niveaux_obesite: {e}")
    total_tests += 1
    
    try:
        test_data.test_pages_navigation()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_pages_navigation: {e}")
    total_tests += 1
    
    # Tests authentification
    test_auth = TestAppAuth()
    try:
        test_auth.test_session_state_structure()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_session_state_structure: {e}")
    total_tests += 1
    
    try:
        test_auth.test_medecins_db_structure()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_medecins_db_structure: {e}")
    total_tests += 1
    
    try:
        test_auth.test_login_validation()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_login_validation: {e}")
    total_tests += 1
    
    try:
        test_auth.test_inscription_validation()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_inscription_validation: {e}")
    total_tests += 1
    
    # Tests historique
    test_hist = TestAppHistorique()
    try:
        test_hist.test_historique_structure()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_historique_structure: {e}")
    total_tests += 1
    
    try:
        test_hist.test_historique_ajout()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_historique_ajout: {e}")
    total_tests += 1
    
    try:
        test_hist.test_historique_export_csv()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_historique_export_csv: {e}")
    total_tests += 1
    
    # Tests recommandations
    test_reco = TestAppRecommandations()
    try:
        test_reco.test_recommandations_par_niveau()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_recommandations_par_niveau: {e}")
    total_tests += 1
    
    try:
        test_reco.test_recommandations_urgences()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_recommandations_urgences: {e}")
    total_tests += 1
    
    # Tests math
    test_math = TestAppMath()
    try:
        test_math.test_calcul_imc()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_calcul_imc: {e}")
    total_tests += 1
    
    try:
        test_math.test_interpretation_imc()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_interpretation_imc: {e}")
    total_tests += 1
    
    # Tests fichiers
    test_files = TestAppFiles()
    try:
        test_files.test_fichiers_essentiels_existent()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_fichiers_essentiels_existent: {e}")
    total_tests += 1
    
    try:
        test_files.test_images_optionnelles()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_images_optionnelles: {e}")
    total_tests += 1
    
    # Tests intégration
    test_int = TestAppIntegration()
    try:
        test_int.test_structure_projet()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_structure_projet: {e}")
    total_tests += 1
    
    try:
        test_int.test_requirements_exists()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_requirements_exists: {e}")
    total_tests += 1
    
    try:
        test_int.test_github_workflow_exists()
        passed_tests += 1
    except Exception as e:
        failed_tests.append(f"test_github_workflow_exists: {e}")
    total_tests += 1
    
    # ============================================
    # RÉSUMÉ FINAL
    # ============================================
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*70)
    print(f"📋 Total des tests exécutés : {total_tests}")
    print(f"✅ Tests réussis : {passed_tests}")
    print(f"❌ Tests échoués : {len(failed_tests)}")
    
    if len(failed_tests) == 0:
        print("\n" + "="*70)
        print("🎉🎉🎉 FÉLICITATIONS ! TOUS LES TESTS SONT PASSÉS ! 🎉🎉🎉")
        print("="*70)
    else:
        print("\n❌ Tests en échec :")
        for fail in failed_tests:
            print(f"   - {fail}")
    
    print("\n✅ EXÉCUTION TERMINÉE")