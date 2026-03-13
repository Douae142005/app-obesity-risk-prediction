# ============================================
# IMPORTATION DES BIBLIOTHÈQUES
# ============================================
import streamlit as st      # Interface web
import pandas as pd         # Manipulation des données
import numpy as np          # Calculs mathématiques
import joblib               # Charger le modèle
import os                   # Gestion des fichiers
import sys                  # Gestion des chemins
import shap                 # Explicabilité du modèle
import matplotlib.pyplot as plt  # Graphiques

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
# Cette partie configure l'apparence générale
# de l'application dans le navigateur
st.set_page_config(
    page_title="ObesityAI",       # Titre dans l'onglet
    page_icon="🏥",               # Icône dans l'onglet
    layout="wide",                # Page en pleine largeur
    initial_sidebar_state="expanded"  # Sidebar ouverte
)

# ============================================
# CHEMINS DES FICHIERS
# ============================================
# On indique où se trouvent les fichiers
# src/ → pour data_processing.py de Meryem
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'src'))

# models/ → pour les fichiers .pkl de Douaa
MODELS_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'models')

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================
# @st.cache_resource → charge une seule fois
# pour ne pas ralentir l'application
@st.cache_resource
def charger_modele():
    """
    Charge les 3 fichiers nécessaires :
    - best_model.pkl  → le modèle ML entraîné
    - scaler.pkl      → normalisation des données
    - label_encoders.pkl → encodage texte→chiffres
    """
    modele    = joblib.load(
        os.path.join(MODELS_DIR, 'best_model.pkl'))
    scaler    = joblib.load(
        os.path.join(MODELS_DIR, 'scaler.pkl'))
    encodeurs = joblib.load(
        os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    return modele, scaler, encodeurs

# Charger le modèle au démarrage de l'app
modele, scaler, encodeurs = charger_modele()
# ============================================
# DESIGN CSS
# ============================================
# Personnalisation visuelle de l'application
# avec des couleurs, polices et cartes stylisées
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

/* Police générale */
* { font-family: 'Inter', sans-serif; }

/* Fond de la page */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
}

/* Section hero (bandeau principal) */
.hero-section {
    background: linear-gradient(135deg, #1a3c5e 0%, #2980b9 50%, #16a085 100%);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(26,60,94,0.3);
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    color: white;
}
.hero-subtitle {
    font-size: 1.2rem;
    color: rgba(255,255,255,0.85);
    margin-top: 0.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    margin-top: 1rem;
    border: 1px solid rgba(255,255,255,0.3);
}

/* Cartes statistiques en haut */
.stat-card {
    background: white;
    border-radius: 15px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-top: 4px solid #2980b9;
}
.stat-number {
    font-size: 2rem;
    font-weight: 800;
    color: #1a3c5e;
}
.stat-label { font-size: 0.85rem; color: #777; }

/* Cartes du formulaire patient */
.form-card {
    background: white;
    border-radius: 18px;
    padding: 1.8rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
    border-left: 5px solid #2980b9;
}
.form-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a3c5e;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #f0f4f8;
}

/* Bouton analyser */
.stButton>button {
    background: linear-gradient(135deg, #1a3c5e, #2980b9);
    color: white !important;
    border-radius: 15px;
    padding: 1rem 3rem;
    font-size: 1.2rem;
    font-weight: 700;
    width: 100%;
    border: none;
    box-shadow: 0 5px 20px rgba(41,128,185,0.4);
}

/* Carte résultat */
.result-card {
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}
.result-title {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0.5rem 0;
}

/* Sidebar (menu gauche) */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3c5e 0%, #2c3e50 100%);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER PRINCIPAL
# ============================================
# Bandeau bleu en haut de la page avec
# le titre et le badge de l'école
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🏥 ObesityAI</div>
    <div class="hero-subtitle">
        Outil d'aide à la décision médicale basé sur l'IA
    </div>
    <div class="hero-badge">
        🎓 Centrale Casablanca – Coding Week 2026
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# STATISTIQUES RAPIDES
# ============================================
# 5 cartes en haut qui affichent les chiffres
# clés du projet
c1, c2, c3, c4, c5 = st.columns(5)
stats = [
    ("2111", "Patients",   "👥"),
    ("17",   "Variables",  "📊"),
    ("7",    "Niveaux",    "🎯"),
    ("3",    "Modèles ML", "🤖"),
    ("96%+", "Précision",  "✅"),
]
for col, (num, label, icon) in zip([c1,c2,c3,c4,c5], stats):
    col.markdown(f"""
    <div class="stat-card">
        <div style="font-size:1.5rem">{icon}</div>
        <div class="stat-number">{num}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# SIDEBAR - NAVIGATION
# ============================================
# Menu de navigation à gauche
# page = la page sélectionnée par l'utilisateur
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:3rem">🏥</div>
        <div style="font-size:1.3rem; font-weight:700;">
            ObesityAI
        </div>
        <div style="font-size:0.8rem;
                    color:rgba(255,255,255,0.6);">
            Aide à la décision médicale
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("📌 Navigation", [
        "🏠 Accueil",
        "👤 Analyse Patient",
        "📊 Statistiques",
        "👥 Notre Équipe",
        "ℹ️ À propos"
    ])

# ============================================
# PAGE : ACCUEIL
# ============================================
if page == "🏠 Accueil":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 👋 Bienvenue sur ObesityAI !")
        st.markdown("""
        **ObesityAI** est un outil clinique intelligent
        qui aide les médecins à estimer le
        **risque d'obésité** grâce au Machine Learning.
        """)
        st.markdown("### 🔄 Comment ça marche ?")
        # Les 4 étapes du fonctionnement
        etapes = [
            ("1️⃣", "Saisir les données du patient",
             "Âge, poids, taille, habitudes..."),
            ("2️⃣", "Analyse par IA",
             "3 modèles ML analysent les données"),
            ("3️⃣", "Résultat instantané",
             "Niveau d'obésité prédit avec confiance"),
            ("4️⃣", "Explication SHAP",
             "Comprendre pourquoi cette prédiction"),
        ]
        for icon, titre, desc in etapes:
            st.markdown(f"""
            <div style="background:#f8f9fa; border-radius:10px;
                        padding:0.8rem 1rem; margin:0.3rem 0;
                        border-left:4px solid #2980b9;">
                <b>{icon} {titre}</b><br>
                <span style="color:#777; font-size:0.9rem;">
                    {desc}
                </span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 Les 7 niveaux")
        # Affichage des 7 classes d'obésité
        niveaux_info = [
            ("🔵", "Insufficient Weight"),
            ("🟢", "Normal Weight"),
            ("🟡", "Overweight Level I"),
            ("🟠", "Overweight Level II"),
            ("🔴", "Obesity Type I"),
            ("🔴", "Obesity Type II"),
            ("🔴", "Obesity Type III"),
        ]
        for emoji, niveau in niveaux_info:
            st.markdown(f"{emoji} {niveau}")

# ============================================
# PAGE : ANALYSE PATIENT
# ============================================
elif page == "👤 Analyse Patient":
    st.markdown("## 👤 Analyse du Patient")
    col1, col2 = st.columns(2)

    # ── Colonne gauche ───────────────────────
    with col1:
        # Carte 1 : Informations physiques
        st.markdown("""
        <div class="form-card">
        <div class="form-card-title">
            👤 Informations physiques
        </div>
        """, unsafe_allow_html=True)
        genre  = st.selectbox("Genre", ["Male", "Female"])
        age    = st.slider("Âge", 10, 80, 25)
        taille = st.slider("Taille (m)", 1.40, 2.10, 1.70, 0.01)
        poids  = st.slider("Poids (kg)", 30, 200, 70)

        # Calcul automatique de l'IMC
        imc = poids / (taille ** 2)
        if imc < 18.5:
            st.info(f"📊 IMC : **{imc:.1f}** — Poids insuffisant")
        elif imc < 25:
            st.success(f"📊 IMC : **{imc:.1f}** — Poids normal ✅")
        elif imc < 30:
            st.warning(f"📊 IMC : **{imc:.1f}** — Surpoids ⚠️")
        else:
            st.error(f"📊 IMC : **{imc:.1f}** — Obésité 🔴")
        st.markdown('</div>', unsafe_allow_html=True)

        # Carte 2 : Habitudes alimentaires
        st.markdown("""
        <div class="form-card">
        <div class="form-card-title">
            🍔 Habitudes alimentaires
        </div>
        """, unsafe_allow_html=True)
        antecedents = st.selectbox(
            "Antécédents familiaux d'obésité", ["yes", "no"])
        favc = st.selectbox("Fast-food fréquent", ["yes", "no"])
        fcvc = st.slider("Fréquence légumes (1-3)", 1.0, 3.0, 2.0, 0.1)
        ncp  = st.slider("Repas par jour", 1.0, 4.0, 3.0, 0.5)
        caec = st.selectbox("Grignotage entre repas",
            ["no", "Sometimes", "Frequently", "Always"])
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Colonne droite ───────────────────────
    with col2:
        # Carte 3 : Activité physique
        st.markdown("""
        <div class="form-card">
        <div class="form-card-title">
            🏃 Activité & Mode de vie
        </div>
        """, unsafe_allow_html=True)
        faf    = st.slider("Sport (jours/semaine)", 0.0, 3.0, 1.0, 0.5)
        tue    = st.slider("Temps écran (h/jour)", 0.0, 2.0, 1.0, 0.25)
        ch2o   = st.slider("Eau (litres/jour)", 1.0, 3.0, 2.0, 0.1)
        mtrans = st.selectbox("Mode de transport", [
            "Public_Transportation", "Walking",
            "Automobile", "Motorbike", "Bike"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Carte 4 : Autres habitudes
        st.markdown("""
        <div class="form-card">
        <div class="form-card-title">
            🚬 Autres habitudes
        </div>
        """, unsafe_allow_html=True)
        smoke = st.selectbox("Fumeur", ["no", "yes"])
        calc  = st.selectbox("Consommation d'alcool",
            ["no", "Sometimes", "Frequently", "Always"])
        scc   = st.selectbox("Surveille ses calories", ["no", "yes"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bouton Analyser ──────────────────────
    if st.button("🔍 Lancer l'analyse IA"):

        # Étape 1 : Créer le DataFrame du patient
        donnees = {
            'Gender': genre,
            'Age': float(age),
            'Height': float(taille),
            'Weight': float(poids),
            'family_history_with_overweight': antecedents,
            'FAVC': favc,
            'FCVC': float(fcvc),
            'NCP': float(ncp),
            'CAEC': caec,
            'SMOKE': smoke,
            'CH2O': float(ch2o),
            'SCC': scc,
            'FAF': float(faf),
            'TUE': float(tue),
            'CALC': calc,
            'MTRANS': mtrans
        }
        df_patient = pd.DataFrame([donnees])

        # Étape 2 : Encoder les colonnes texte
        # (convertir Male→1, Female→0, etc.)
        for col_name, le in encodeurs.items():
            if col_name in df_patient.columns:
                df_patient[col_name] = le.transform(
                    df_patient[col_name].astype(str))

      # Colonnes numériques seulement pour le scaler
        cols_numeriques = [
            'Age', 'Height', 'Weight',
            'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'
        ]

        # Colonnes catégorielles encodées
        cols_categorielles = [
            'Gender', 'family_history_with_overweight',
            'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
        ]

        # Normaliser seulement les colonnes numériques
        df_num = df_patient[cols_numeriques].astype(float)
        df_num_sc = pd.DataFrame(
            scaler.transform(df_num),
            columns=cols_numeriques)

        # Colonnes catégorielles encodées
        df_cat = df_patient[cols_categorielles].astype(float)

        # Combiner les deux dans le bon ordre
        df_sc = pd.concat([df_cat, df_num_sc], axis=1)

        # Réorganiser dans l'ordre du modèle
        colonnes_finales = [
            'Gender', 'Age', 'Height', 'Weight',
            'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'CAEC',
            'SMOKE', 'CH2O', 'SCC', 'FAF',
            'TUE', 'CALC', 'MTRANS'
        ]
        df_sc = df_sc[colonnes_finales]

        # Étape 6 : Prédire avec le modèle
        prediction = modele.predict(df_sc)[0]
        probas     = modele.predict_proba(df_sc)[0]

        # Dictionnaire des 7 niveaux d'obésité
        # (index → emoji, label, couleur)
        niveaux = {
            0: ("🔵", "Insufficient Weight",  "#3498db"),
            1: ("🟢", "Normal Weight",        "#2ecc71"),
            2: ("🟡", "Overweight Level I",   "#f1c40f"),
            3: ("🟠", "Overweight Level II",  "#e67e22"),
            4: ("🔴", "Obesity Type I",       "#e74c3c"),
            5: ("🔴", "Obesity Type II",      "#c0392b"),
            6: ("🔴", "Obesity Type III",     "#922b21"),
        }

        emoji, label, couleur = niveaux[prediction]

        # ── Affichage du résultat ────────────
        st.markdown("---")
        st.markdown("### 🎯 Résultat de l'analyse")
        st.markdown(f"""
        <div class="result-card"
             style="background:{couleur}22;
                    border:3px solid {couleur};">
            <div style="font-size:3rem">{emoji}</div>
            <div class="result-title"
                 style="color:{couleur};">{label}</div>
            <div style="color:#555;">
                Confiance : {probas[prediction]*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probabilités par niveau ──────────
        st.markdown("### 📊 Probabilités par niveau")
        for idx, (em, lbl, col) in niveaux.items():
            st.progress(
                float(probas[idx]),
                text=f"{em} {lbl} : {probas[idx]*100:.1f}%")

        # ── Explication SHAP ─────────────────
        # SHAP explique POURQUOI le modèle
        # a prédit ce niveau d'obésité
        st.markdown("---")
        st.markdown("### 🔬 Explication SHAP")
        st.markdown("""
        *Quelles variables ont le plus influencé
        la prédiction ?*
        """)
        try:
            # Créer l'explainer SHAP
            explainer = shap.TreeExplainer(modele)

            # Calculer les valeurs SHAP
            shap_values = explainer.shap_values(df_sc)

            # Graphique SHAP
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(
                shap_values,
                df_sc,
                plot_type="bar",
                show=False
            )
            plt.title("Impact des variables sur la prédiction")
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.warning(f"⚠️ SHAP non disponible : {e}")