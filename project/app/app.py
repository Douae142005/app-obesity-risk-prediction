import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ObesityAI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def charger_modele():
    modele    = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    scaler    = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    encodeurs = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    return modele, scaler, encodeurs

modele, scaler, encodeurs = charger_modele()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%); }
.hero-section {
    background: linear-gradient(135deg, #1a3c5e 0%, #2980b9 50%, #16a085 100%);
    border-radius: 20px; padding: 3rem 2rem; text-align: center;
    margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(26,60,94,0.3);
}
.hero-title { font-size: 3.5rem; font-weight: 800; color: white; }
.hero-subtitle { font-size: 1.2rem; color: rgba(255,255,255,0.85); margin-top: 0.5rem; }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.2); color: white;
    padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.85rem;
    margin-top: 1rem; border: 1px solid rgba(255,255,255,0.3);
}
.stat-card {
    background: white; border-radius: 15px; padding: 1.2rem;
    text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-top: 4px solid #2980b9;
}
.stat-number { font-size: 2rem; font-weight: 800; color: #1a3c5e; }
.stat-label { font-size: 0.85rem; color: #777; }
.form-card {
    background: white; border-radius: 18px; padding: 1.8rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 1rem; border-left: 5px solid #2980b9;
}
.form-card-title {
    font-size: 1.1rem; font-weight: 700; color: #1a3c5e;
    margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid #f0f4f8;
}
.stButton>button {
    background: linear-gradient(135deg, #1a3c5e, #2980b9);
    color: white !important; border-radius: 15px; padding: 1rem 3rem;
    font-size: 1.2rem; font-weight: 700; width: 100%; border: none;
    box-shadow: 0 5px 20px rgba(41,128,185,0.4);
}
.result-card {
    border-radius: 18px; padding: 2rem; text-align: center;
    margin: 1rem 0; box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}
.result-title { font-size: 1.8rem; font-weight: 800; margin: 0.5rem 0; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3c5e 0%, #2c3e50 100%);
}
section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <div class="hero-title">🏥 ObesityAI</div>
    <div class="hero-subtitle">Outil d'aide à la décision médicale basé sur l'IA</div>
    <div class="hero-badge">🎓 Centrale Casablanca – Coding Week 2026</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
stats = [
    ("2111", "Patients", "👥"),
    ("17", "Variables", "📊"),
    ("7", "Niveaux", "🎯"),
    ("3", "Modèles ML", "🤖"),
    ("96%+", "Précision", "✅"),
]
for col, (num, label, icon) in zip([c1, c2, c3, c4, c5], stats):
    col.markdown(f"""
    <div class="stat-card">
        <div style="font-size:1.5rem">{icon}</div>
        <div class="stat-number">{num}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:3rem">🏥</div>
        <div style="font-size:1.3rem; font-weight:700;">ObesityAI</div>
        <div style="font-size:0.8rem; color:rgba(255,255,255,0.6);">
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

if page == "🏠 Accueil":
    st.markdown("## 👋 Bienvenue sur ObesityAI !")
    st.markdown("""
    **ObesityAI** aide les médecins à estimer le
    **risque d'obésité** grâce au Machine Learning.
    """)

elif page == "👤 Analyse Patient":
    st.markdown("## 👤 Analyse du Patient")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="form-card"><div class="form-card-title">👤 Informations physiques</div>', unsafe_allow_html=True)
        genre  = st.selectbox("Genre", ["Male", "Female"])
        age    = st.slider("Âge", 10, 80, 25)
        taille = st.slider("Taille (m)", 1.40, 2.10, 1.70, 0.01)
        poids  = st.slider("Poids (kg)", 30, 200, 70)
        imc    = poids / (taille ** 2)
        st.info(f"📊 IMC : **{imc:.1f}**")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-card"><div class="form-card-title">🍔 Habitudes alimentaires</div>', unsafe_allow_html=True)
        antecedents = st.selectbox("Antécédents familiaux", ["yes", "no"])
        favc = st.selectbox("Fast-food fréquent", ["yes", "no"])
        fcvc = st.slider("Fréquence légumes (1-3)", 1.0, 3.0, 2.0, 0.1)
        ncp  = st.slider("Repas par jour", 1.0, 4.0, 3.0, 0.5)
        caec = st.selectbox("Grignotage", ["no", "Sometimes", "Frequently", "Always"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="form-card"><div class="form-card-title">🏃 Activité & Mode de vie</div>', unsafe_allow_html=True)
        faf    = st.slider("Sport (jours/semaine)", 0.0, 3.0, 1.0, 0.5)
        tue    = st.slider("Temps écran (h/jour)", 0.0, 2.0, 1.0, 0.25)
        ch2o   = st.slider("Eau (litres/jour)", 1.0, 3.0, 2.0, 0.1)
        mtrans = st.selectbox("Transport", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-card"><div class="form-card-title">🚬 Autres habitudes</div>', unsafe_allow_html=True)
        smoke = st.selectbox("Fumeur", ["no", "yes"])
        calc  = st.selectbox("Alcool", ["no", "Sometimes", "Frequently", "Always"])
        scc   = st.selectbox("Surveille calories", ["no", "yes"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Lancer l'analyse IA"):
        donnees = {
            'Gender': genre, 'Age': age,
            'Height': taille, 'Weight': poids,
            'family_history_with_overweight': antecedents,
            'FAVC': favc, 'FCVC': fcvc, 'NCP': ncp,
            'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o,
            'SCC': scc, 'FAF': faf, 'TUE': tue,
            'CALC': calc, 'MTRANS': mtrans
        }

        df_patient = pd.DataFrame([donnees])

        for col_name, le in encodeurs.items():
            if col_name in df_patient.columns:
                df_patient[col_name] = le.transform(
                    df_patient[col_name].astype(str))

        df_sc = pd.DataFrame(
            scaler.transform(df_patient),
            columns=df_patient.columns)

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
        st.markdown("### 🎯 Résultat")
        st.markdown(f"""
        <div class="result-card"
             style="background:{couleur}22; border:3px solid {couleur};">
            <div style="font-size:3rem">{emoji}</div>
            <div class="result-title" style="color:{couleur};">{label}</div>
            <div style="color:#555;">Confiance : {probas[prediction]*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Probabilités")
        for idx, (em, lbl, col) in niveaux.items():
            st.progress(
                float(probas[idx]),
                text=f"{em} {lbl} : {probas[idx]*100:.1f}%")

        st.markdown("---")
        st.markdown("### 🔬 Explication SHAP")
        st.markdown("*Quelles variables ont influencé la prédiction ?*")
        try:
            explainer   = shap.TreeExplainer(modele)
            shap_values = explainer.shap_values(df_sc)
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values, df_sc,
                              plot_type="bar", show=False)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"⚠️ SHAP : {e}")

elif page == "📊 Statistiques":
    st.markdown("## 📊 Statistiques")
    st.info("⏳ Disponible prochainement...")

elif page == "👥 Notre Équipe":
    st.markdown("## 👥 Notre Équipe")
    membres = [
        ("👩‍💻", "Meryem",   "Data Processing", "#3498db"),
        ("👩‍💻", "Amina",    "Analyse EDA",      "#2ecc71"),
        ("👩‍💻", "Douaa",    "Modèles ML",       "#e74c3c"),
        ("👩‍💻", "Hajar AZ", "SHAP & Évaluation","#f39c12"),
        ("👩‍💻", "Hajar D",  "Interface",        "#9b59b6"),
    ]
    cols = st.columns(5)
    for col, (icon, nom, role, color) in zip(cols, membres):
        col.markdown(f"""
        <div style="background:white; border-radius:15px;
                    padding:1.2rem; text-align:center;
                    box-shadow:0 4px 15px rgba(0,0,0,0.08);
                    border-top:4px solid {color};">
            <div style="font-size:2.5rem">{icon}</div>
            <div style="font-weight:700; color:#1a3c5e;">{nom}</div>
            <div style="color:{color}; font-size:0.85rem;">{role}</div>
        </div>
        """, unsafe_allow_html=True)

elif page == "ℹ️ À propos":
    st.markdown("## ℹ️ À propos")
    st.markdown("""
    - **Dataset** : UCI id=544 – 2111 patients
    - **Modèles** : Random Forest, XGBoost, LightGBM
    - **Explicabilité** : SHAP
    - **Deadline** : 15 Mars 2026
    """)