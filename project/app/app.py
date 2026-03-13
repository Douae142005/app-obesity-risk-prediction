# ============================================
# IMPORTATION DES BIBLIOTHÈQUES
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import shap
import matplotlib.pyplot as plt
from PIL import Image

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
st.set_page_config(
    page_title="NutriScan AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CHEMINS DES FICHIERS
# ============================================
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'src'))
MODELS_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'models')
APP_DIR = os.path.dirname(__file__)

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================
@st.cache_resource
def charger_modele():
    modele    = joblib.load(
        os.path.join(MODELS_DIR, 'best_model.pkl'))
    scaler    = joblib.load(
        os.path.join(MODELS_DIR, 'scaler.pkl'))
    encodeurs = joblib.load(
        os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    return modele, scaler, encodeurs

modele, scaler, encodeurs = charger_modele()

# ============================================
# DESIGN CSS
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
}

/* Hero avec médecin en arrière-plan */
.hero-section {
    background: linear-gradient(135deg, 
        rgba(26,60,94,0.92) 0%, 
        rgba(41,128,185,0.85) 50%, 
        rgba(22,160,133,0.90) 100%),
        url('https://img.freepik.com/free-photo/doctor-with-stethoscope-hands-hospital-background_1423-1.jpg');
    background-size: cover;
    background-position: center;
    border-radius: 20px;
    padding: 4rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(26,60,94,0.3);
}
.hero-logo {
    width: 80px;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    color: white;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.hero-subtitle {
    font-size: 1.2rem;
    color: rgba(255,255,255,0.90);
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

.stat-card {
    background: white;
    border-radius: 15px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-top: 4px solid #2980b9;
}
.stat-number { font-size: 2rem; font-weight: 800; color: #1a3c5e; }
.stat-label { font-size: 0.85rem; color: #777; }

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

.result-card {
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}
.result-title { font-size: 1.8rem; font-weight: 800; margin: 0.5rem 0; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3c5e 0%, #2c3e50 100%);
}
section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER PRINCIPAL avec médecin en arrière-plan
# ============================================
# Convertir les images en base64 pour le slideshow
import base64

def img_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

logo_b64 = img_to_base64(os.path.join(APP_DIR, 'logo.png'))
bg1_b64  = img_to_base64(os.path.join(APP_DIR, 'bg1.jpg'))
bg2_b64  = img_to_base64(os.path.join(APP_DIR, 'bg2.jpg'))
bg3_b64  = img_to_base64(os.path.join(APP_DIR, 'bg3.jpg'))
bg4_b64  = img_to_base64(os.path.join(APP_DIR, 'bg4.jpg'))

st.markdown(f"""
<style>
.slideshow {{
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    height: 300px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(26,60,94,0.3);
}}
.slide {{
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-size: cover;
    background-position: center;
    opacity: 0;
    animation: slideshow 16s infinite;
}}
.slide:nth-child(1) {{
    background-image: url('data:image/jpeg;base64,{bg1_b64}');
    animation-delay: 0s;
}}
.slide:nth-child(2) {{
    background-image: url('data:image/jpeg;base64,{bg2_b64}');
    animation-delay: 4s;
}}
.slide:nth-child(3) {{
    background-image: url('data:image/jpeg;base64,{bg3_b64}');
    animation-delay: 8s;
}}
.slide:nth-child(4) {{
    background-image: url('data:image/jpeg;base64,{bg4_b64}');
    animation-delay: 12s;
}}
@keyframes slideshow {{
    0%   {{ opacity: 0; }}
    5%   {{ opacity: 1; }}
    25%  {{ opacity: 1; }}
    30%  {{ opacity: 0; }}
    100% {{ opacity: 0; }}
}}
.hero-overlay {{
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(
        135deg,
        rgba(26,60,94,0.75) 0%,
        rgba(41,128,185,0.65) 50%,
        rgba(22,160,133,0.70) 100%
    );
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
}}
.hero-title {{
    font-size: 3rem;
    font-weight: 800;
    color: white;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
    margin: 0.5rem 0;
}}
.hero-subtitle {{
    font-size: 1.1rem;
    color: rgba(255,255,255,0.90);
    margin-top: 0.3rem;
}}
.hero-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    margin-top: 0.8rem;
    border: 1px solid rgba(255,255,255,0.3);
}}
</style>

<div class="slideshow">
    <div class="slide"></div>
    <div class="slide"></div>
    <div class="slide"></div>
    <div class="slide"></div>
    <div class="hero-overlay">
        <img src="data:image/png;base64,{logo_b64}"
             style="width:80px; margin-bottom:0.5rem;
                    border-radius:10px;">
        <div class="hero-title">🏥 NutriScan AI</div>
        <div class="hero-subtitle">
            Prédiction intelligente du risque d'obésité par LightGBM
        </div>
        <div class="hero-badge">
            🎓 Centrale Casablanca – Coding Week 2026
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# STATISTIQUES RAPIDES
# ============================================
c1, c2, c3, c4, c5 = st.columns(5)
stats = [
    ("2111", "Patients",  "👥"),
    ("17",   "Variables", "📊"),
    ("7",    "Niveaux",   "🎯"),
    ("1",    "Modèle ML", "🤖"),
    ("97%+", "Précision", "✅"),
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
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <img src="https://cdn-icons-png.flaticon.com/512/2977/2977339.png"
             style="width:60px; margin-bottom:0.5rem;">
        <div style="font-size:1.3rem; font-weight:700;">
            NutriScan AI
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
        st.markdown("## 👋 Bienvenue sur NutriScan AI !")
        st.markdown("""
        **NutriScan AI** est un outil clinique intelligent
        qui aide les médecins à estimer le
        **risque d'obésité** grâce au modèle **LightGBM**.
        """)
        st.markdown("### 🔄 Comment ça marche ?")
        etapes = [
            ("1️⃣", "Saisir les données du patient",
             "Âge, poids, taille, habitudes..."),
            ("2️⃣", "Analyse par LightGBM",
             "Le modèle analyse les 16 variables"),
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
        # Logo santé
        st.markdown("""
        <div style="text-align:center; padding:1rem;">
            <img src="https://cdn-icons-png.flaticon.com/512/2977/2977339.png"
                 style="width:120px;">
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 🎯 Les 7 niveaux")
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

    with col1:
        st.markdown("""
        <div class="form-card">
        <div class="form-card-title">👤 Informations physiques</div>
        """, unsafe_allow_html=True)
        # Champs de saisie au lieu de sliders
        genre  = st.selectbox("Genre", ["Male", "Female"])
        age    = st.number_input("Âge", min_value=10, max_value=80, value=25)
        taille = st.number_input("Taille (m)", min_value=1.40, max_value=2.10, value=1.70, step=0.01, format="%.2f")
        poids  = st.number_input("Poids (kg)", min_value=30, max_value=200, value=70)

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

        st.markdown("""
        <div class="form-card">
        <div class="form-card-title">🍔 Habitudes alimentaires</div>
        """, unsafe_allow_html=True)
        antecedents = st.selectbox(
            "Antécédents familiaux d'obésité", ["yes", "no"])
        favc = st.selectbox("Fast-food fréquent", ["yes", "no"])
        fcvc = st.number_input("Fréquence légumes (1-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1, format="%.1f")
        ncp  = st.number_input("Repas par jour", min_value=1.0, max_value=4.0, value=3.0, step=0.5, format="%.1f")
        caec = st.selectbox("Grignotage entre repas",
            ["no", "Sometimes", "Frequently", "Always"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="form-card">
        <div class="form-card-title">🏃 Activité & Mode de vie</div>
        """, unsafe_allow_html=True)
        faf  = st.number_input("Sport (jours/semaine)", min_value=0.0, max_value=3.0, value=1.0, step=0.5, format="%.1f")
        tue  = st.number_input("Temps écran (h/jour)", min_value=0.0, max_value=2.0, value=1.0, step=0.25, format="%.2f")
        ch2o = st.number_input("Eau (litres/jour)", min_value=1.0, max_value=3.0, value=2.0, step=0.1, format="%.1f")
        mtrans = st.selectbox("Mode de transport", [
            "Public_Transportation", "Walking",
            "Automobile", "Motorbike", "Bike"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="form-card">
        <div class="form-card-title">🚬 Autres habitudes</div>
        """, unsafe_allow_html=True)
        smoke = st.selectbox("Fumeur", ["no", "yes"])
        calc  = st.selectbox("Consommation d'alcool",
            ["no", "Sometimes", "Frequently", "Always"])
        scc   = st.selectbox("Surveille ses calories", ["no", "yes"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Lancer l'analyse IA"):
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

        for col_name, le in encodeurs.items():
            if col_name in df_patient.columns:
                df_patient[col_name] = le.transform(
                    df_patient[col_name].astype(str))

        cols_numeriques = [
            'Age', 'Height', 'Weight',
            'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'
        ]
        cols_categorielles = [
            'Gender', 'family_history_with_overweight',
            'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
        ]

        df_num    = df_patient[cols_numeriques].astype(float)
        df_num_sc = pd.DataFrame(
            scaler.transform(df_num),
            columns=cols_numeriques)
        df_cat = df_patient[cols_categorielles].astype(float)
        df_sc  = pd.concat([df_cat, df_num_sc], axis=1)

        colonnes_finales = [
            'Gender', 'Age', 'Height', 'Weight',
            'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'CAEC',
            'SMOKE', 'CH2O', 'SCC', 'FAF',
            'TUE', 'CALC', 'MTRANS'
        ]
        df_sc = df_sc[colonnes_finales]

        prediction = modele.predict(df_sc)[0]
        probas     = modele.predict_proba(df_sc)[0]

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

        st.markdown("### 📊 Probabilités par niveau")
        for idx, (em, lbl, col) in niveaux.items():
            st.progress(
                float(probas[idx]),
                text=f"{em} {lbl} : {probas[idx]*100:.1f}%")

        st.markdown("---")
        st.markdown("### 🔬 Explication SHAP")
        st.markdown("*Quelles variables ont le plus influencé la prédiction ?*")
        try:
            explainer   = shap.TreeExplainer(modele)
            shap_values = explainer.shap_values(df_sc)
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values, df_sc,
                              plot_type="bar", show=False)
            plt.title("Impact des variables sur la prédiction")
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"⚠️ SHAP non disponible : {e}")

# ============================================
# PAGE : STATISTIQUES
# ============================================
elif page == "📊 Statistiques":
    st.markdown("## 📊 Statistiques du Dataset")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=544)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)

        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Patients", "2111")
        col2.metric("📊 Variables", "17")
        col3.metric("🎯 Classes", "7")

        st.markdown("### Distribution des niveaux d'obésité")
        fig, ax = plt.subplots(figsize=(10, 4))
        y.value_counts().plot(
            kind='bar', ax=ax, color='#2980b9')
        ax.set_xlabel("Niveau d'obésité")
        ax.set_ylabel("Nombre de patients")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"⚠️ Dataset non disponible : {e}")

# ============================================
# PAGE : NOTRE ÉQUIPE
# ============================================
elif page == "👥 Notre Équipe":
    st.markdown("## 👥 Notre Équipe")
    st.markdown("### 🤝 Un projet réalisé en groupe !")

    col_img, col_noms = st.columns([2, 1])

    with col_img:
        # Cherche la photo dans app/team.jpg
        team_path = os.path.join(APP_DIR, 'team.jpg')
        if os.path.exists(team_path):
            st.image(team_path,
                     caption="Notre équipe – Coding Week 2026",
                     use_column_width=True)
        else:
            st.info("📸 Ajoutez votre photo dans app/team.jpg")

    with col_noms:
        st.markdown("### 👩‍💻 Les membres")
        membres = [
            ("Meryem",   "#3498db"),
            ("Amina",    "#2ecc71"),
            ("Douaa",    "#e74c3c"),
            ("Hajar AZ", "#f39c12"),
            ("Hajar D",  "#9b59b6"),
        ]
        for nom, color in membres:
            st.markdown(f"""
            <div style="background:white; border-radius:10px;
                        padding:0.8rem 1rem; margin:0.4rem 0;
                        box-shadow:0 3px 10px rgba(0,0,0,0.08);
                        border-left:5px solid {color};">
                <span style="font-weight:700;
                             color:#1a3c5e;">👩‍💻 {nom}</span>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# PAGE : À PROPOS
# ============================================
elif page == "ℹ️ À propos":
    st.markdown("## ℹ️ À propos du projet")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Objectif
        Développer un outil clinique basé sur le ML
        pour estimer le risque d'obésité.

        ### 📁 Dataset UCI
        - **Source** : UCI Machine Learning Repository
        - **ID** : 544
        - **Patients** : 2111
        - **Variables** : 17
        - **Classes** : 7
        """)

    with col2:
        st.markdown("""
        ### 🛠️ Technologies
        - **Python** — Langage principal
        - **Streamlit** — Interface web
        - **LightGBM** — Modèle ML (97%+ précision)
        - **SHAP** — Explicabilité
        - **GitHub Actions** — CI/CD
        """)
