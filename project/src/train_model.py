
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
# ============================================
# 3. ENTRAÎNEMENT DES MODÈLES (3 SEULEMENT)
# ============================================

def train_all_models(X_train, X_test, y_train, y_test):
    """Entraîne les 3 modèles demandés."""
    
    results = []
    classes = sorted(y_train.unique())
    
    # 1. RANDOM FOREST
    print("\n" + "="*60)
    print("1️⃣  RANDOM FOREST")
    print("="*60)
    
    try:
        print("   Entraînement en cours...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        print("   ✅ Random Forest entraîné")
        
        result = evaluate_model(rf, X_test, y_test, "Random Forest", classes)
        if result:
            results.append(result)
    except Exception as e:
        print(f"❌ Erreur Random Forest: {e}")
    
    # 2. XGBOOST
    print("\n" + "="*60)
    print("2️⃣  XGBOOST")
    print("="*60)
    
    try:
        from xgboost import XGBClassifier
        print("   ✅ XGBoost importé")
        
        print("   Entraînement en cours...")
        xgb = XGBClassifier(n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False)
        xgb.fit(X_train, y_train)
        print("   ✅ XGBoost entraîné")
        
        result = evaluate_model(xgb, X_test, y_test, "XGBoost", classes)
        if result:
            results.append(result)
    except ImportError:
        print("   ❌ XGBoost non installé - installation recommandée:")
        print("   pip install xgboost")
    except Exception as e:
        print(f"   ❌ Erreur XGBoost: {e}")
    
    # 3. LIGHTGBM
    print("\n" + "="*60)
    print("3️⃣  LIGHTGBM")
    print("="*60)
    
    try:
        from lightgbm import LGBMClassifier
        print("   ✅ LightGBM importé")
        
        print("   Entraînement en cours...")
        lgb = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        lgb.fit(X_train, y_train)
        print("   ✅ LightGBM entraîné")
        
        result = evaluate_model(lgb, X_test, y_test, "LightGBM", classes)
        if result:
            results.append(result)
    except ImportError:
        print("   ❌ LightGBM non installé - installation recommandée:")
        print("   pip install lightgbm")
    except Exception as e:
        print(f"   ❌ Erreur LightGBM: {e}")
    
    return results


# ============================================
# 4. SÉLECTION DU MEILLEUR MODÈLE
# ============================================

def select_best_model(results):
    """Sélectionne le meilleur modèle."""
    
    if not results:
        print("\n❌ Aucun modèle disponible")
        return None
    
    best = max(results, key=lambda x: x['accuracy'])
    
    print("\n" + "="*60)
    print(f"🏆 MEILLEUR MODÈLE: {best['model_name']}")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    print(f"   F1-Score: {best['f1_score']:.4f}")
    print("="*60)
    
    return best
# ============================================
# 5. SAUVEGARDE
# ============================================

def save_best_model(best_result):
    """Sauvegarde le meilleur modèle."""
    
    if not best_result:
        return
    
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, 'best_model.pkl')
        joblib.dump(best_result['model'], model_path)
        print(f"\n✅ Modèle sauvegardé: {model_path}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
# ============================================
# 6. TABLEAU COMPARATIF
# ============================================

def print_comparison_table(results):
    """Affiche un tableau comparatif."""
    
    if not results:
        print("\n❌ Aucun résultat à comparer")
        return
    
    df = pd.DataFrame([{
        'Modèle': r['model_name'],
        'Accuracy': round(r['accuracy'], 4),
        'F1-Score': round(r['f1_score'], 4)
    } for r in results])
    
    print("\n📊 COMPARAISON DES MODÈLES:")
    print(df.to_string(index=False))


# ============================================
# 7. EXÉCUTION PRINCIPALE
# ============================================

if __name__ == "__main__":
    
    try:
        # Prétraitement
        print("\n📥 Chargement des données...")
        X_train, X_test, y_train, y_test, label_encoders, scaler = preprocess_pipeline()
        print("✅ Données prêtes")
        print(f"   Train: {X_train.shape}")
        print(f"   Test: {X_test.shape}")

        # Entraînement
        results = train_all_models(X_train, X_test, y_train, y_test)

        # Résultats
        if results:
            print_comparison_table(results)
            best = select_best_model(results)
            save_best_model(best)
        else:
            print("\n❌ Aucun modèle n'a été entraîné")

    except Exception as e:
        print(f"\n❌ Erreur générale: {e}")

    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*60)