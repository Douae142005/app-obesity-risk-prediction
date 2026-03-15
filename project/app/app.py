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
import base64
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
st.set_page_config(
    page_title="ObesityRisk AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CHEMINS DES FICHIERS
# ============================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
APP_DIR    = os.path.dirname(__file__)

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================
@st.cache_resource
def charger_modele():
    modele    = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    scaler    = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    encodeurs = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    return modele, scaler, encodeurs

modele, scaler, encodeurs = charger_modele()

# ============================================
# IMAGES BASE64
# ============================================
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

# ============================================
# COMPTES MÉDECINS (stockés en session)
# ============================================
if 'medecins_db' not in st.session_state:
    st.session_state.medecins_db = {}

MEDECINS = st.session_state.medecins_db

# ============================================
# SESSION STATE
# ============================================
if 'connecte'      not in st.session_state: st.session_state.connecte      = False
if 'medecin_nom'   not in st.session_state: st.session_state.medecin_nom   = ""
if 'medecin_email' not in st.session_state: st.session_state.medecin_email = ""
if 'historique'    not in st.session_state: st.session_state.historique    = []
if 'page'          not in st.session_state: st.session_state.page          = "🏠 Accueil"

# ============================================
# RECOMMANDATIONS PAR NIVEAU
# ============================================
RECOMMANDATIONS = {
    0: {
        "titre": "⚠️ Poids Insuffisant",
        "couleur": "#3498db",
        "urgence": "Consultation nutritionniste recommandée",
        "conseils": [
            "🍽️ Augmentez vos apports caloriques avec des aliments nutritifs",
            "🥩 Consommez plus de protéines (viandes, légumineuses, œufs)",
            "🏋️ Pratiquez des exercices de renforcement musculaire",
            "🩺 Consultez un nutritionniste pour un plan alimentaire adapté",
            "💊 Vérifiez l'absence de carences en vitamines et minéraux",
        ]
    },
    1: {
        "titre": "✅ Poids Normal — Continuez ainsi !",
        "couleur": "#2ecc71",
        "urgence": "Aucune action urgente nécessaire",
        "conseils": [
            "🥗 Maintenez une alimentation équilibrée et variée",
            "🏃 Continuez l'activité physique régulière (150 min/semaine)",
            "💧 Buvez au moins 2 litres d'eau par jour",
            "😴 Assurez 7-8 heures de sommeil par nuit",
            "📋 Faites un bilan de santé annuel",
        ]
    },
    2: {
        "titre": "🟡 Surpoids Niveau I",
        "couleur": "#f1c40f",
        "urgence": "Suivi médical conseillé dans les 3 mois",
        "conseils": [
            "🥦 Augmentez la consommation de légumes et fruits",
            "🚶 Marchez minimum 30 minutes par jour",
            "🍬 Réduisez les sucres raffinés et aliments transformés",
            "📱 Limitez le temps d'écran à 2h/jour maximum",
            "🩺 Consultez votre médecin pour un suivi régulier",
        ]
    },
    3: {
        "titre": "🟠 Surpoids Niveau II",
        "couleur": "#e67e22",
        "urgence": "Consultation médicale urgente recommandée",
        "conseils": [
            "🍽️ Adoptez un régime hypocalorique supervisé",
            "🏊 Pratiquez une activité physique 3-4 fois par semaine",
            "🚭 Arrêtez de fumer si vous fumez",
            "🍷 Réduisez drastiquement la consommation d'alcool",
            "🩺 Bilan sanguin complet (cholestérol, glycémie)",
            "👨‍⚕️ Consultation avec un endocrinologue recommandée",
        ]
    },
    4: {
        "titre": "🔴 Obésité Type I",
        "couleur": "#e74c3c",
        "urgence": "🚨 Prise en charge médicale immédiate requise",
        "conseils": [
            "👨‍⚕️ Consultez immédiatement un médecin spécialiste",
            "🏥 Programme de perte de poids médical supervisé",
            "💊 Évaluation des comorbidités (diabète, hypertension)",
            "🧘 Thérapie comportementale et cognitive recommandée",
            "🥗 Plan nutritionnel strict avec diététicien",
            "❤️ Surveillance cardiovasculaire régulière",
        ]
    },
    5: {
        "titre": "🔴 Obésité Type II",
        "couleur": "#c0392b",
        "urgence": "🚨 Urgence médicale — Hospitalisation recommandée",
        "conseils": [
            "🚨 Hospitalisation pour bilan complet recommandée",
            "💊 Traitement médicamenteux possible sous supervision",
            "🏥 Programme intensif de réhabilitation",
            "❤️ Surveillance cardiaque et tensionnelle quotidienne",
            "🩸 Contrôle glycémique strict",
            "👨‍⚕️ Équipe pluridisciplinaire (cardiologue, nutritionniste)",
        ]
    },
    6: {
        "titre": "🔴 Obésité Type III — Urgence absolue",
        "couleur": "#922b21",
        "urgence": "🚨 URGENCE ABSOLUE — Intervention médicale immédiate",
        "conseils": [
            "🚨 Consultation chirurgicale bariatrique à envisager",
            "🏥 Prise en charge hospitalière multidisciplinaire",
            "💊 Traitement médicamenteux intensif",
            "❤️ Surveillance cardiaque continue",
            "🧠 Soutien psychologique indispensable",
            "📋 Évaluation pour chirurgie bariatrique",
        ]
    },
}

# ============================================
# CSS 
# ============================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Nunito:wght@300;400;600;700;800&display=swap');
* {{ font-family: 'Nunito', sans-serif; }}

.slideshow {{ position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:-1; overflow:hidden; }}
.slide {{ position:absolute; top:0; left:0; width:100%; height:100%; background-size:cover; background-position:center; opacity:0; animation:slideshow 16s infinite; }}
.slide:nth-child(1) {{ background-image:url('data:image/jpeg;base64,{bg1_b64}'); animation-delay:0s; }}
.slide:nth-child(2) {{ background-image:url('data:image/jpeg;base64,{bg2_b64}'); animation-delay:4s; }}
.slide:nth-child(3) {{ background-image:url('data:image/jpeg;base64,{bg3_b64}'); animation-delay:8s; }}
.slide:nth-child(4) {{ background-image:url('data:image/jpeg;base64,{bg4_b64}'); animation-delay:12s; }}
@keyframes slideshow {{ 0%{{opacity:0}} 5%{{opacity:1}} 25%{{opacity:1}} 30%{{opacity:0}} 100%{{opacity:0}} }}

.stApp {{ background:rgba(5,15,30,0.4); }}

.hero-section {{
    background:linear-gradient(135deg,rgba(15,40,80,0.92) 0%,rgba(20,100,180,0.82) 50%,rgba(10,140,120,0.87) 100%);
    border-radius:24px; padding:3rem 2rem; text-align:center; margin-bottom:2rem;
    box-shadow:0 15px 40px rgba(0,0,0,0.5); border:1px solid rgba(255,255,255,0.1);
}}
.hero-title {{
    font-family:'Rajdhani',sans-serif; font-size:3.5rem; font-weight:700;
    color:white; letter-spacing:4px; text-shadow:0 0 30px rgba(41,182,246,0.6); margin:0.5rem 0;
}}
.hero-subtitle {{ font-size:1.1rem; color:rgba(255,255,255,0.85); }}
.hero-badge {{
    display:inline-block; background:rgba(255,255,255,0.15); color:white;
    padding:0.3rem 1.2rem; border-radius:20px; font-size:0.85rem;
    margin-top:0.8rem; border:1px solid rgba(255,255,255,0.3);
}}

/* SIDEBAR ÉNERGIQUE */
section[data-testid="stSidebar"] {{
    background:linear-gradient(160deg,#0a0f1e 0%,#0d2137 30%,#0a3d2e 60%,#0d1f3c 100%) !important;
    border-right:2px solid rgba(41,182,246,0.25);
    box-shadow:4px 0 25px rgba(0,0,0,0.6);
}}
section[data-testid="stSidebar"]::before {{
    content:''; position:absolute; top:0; left:0; right:0; bottom:0;
    background:repeating-linear-gradient(90deg,transparent,transparent 40px,rgba(41,182,246,0.03) 40px,rgba(41,182,246,0.03) 41px);
    pointer-events:none;
}}
section[data-testid="stSidebar"] * {{ color:white !important; }}

.medecin-badge {{
    background:linear-gradient(135deg,rgba(16,185,129,0.25),rgba(41,182,246,0.15));
    border:1px solid rgba(16,185,129,0.35); border-radius:14px;
    padding:12px 16px; margin:10px 0; text-align:center;
}}
.stat-box {{ background:rgba(255,255,255,0.07); border-radius:10px; padding:8px 12px; margin:4px 0; text-align:center; border:1px solid rgba(255,255,255,0.08); }}
.stat-number {{ font-family:'Rajdhani',sans-serif; font-size:1.8rem; font-weight:700; color:#29b6f6 !important; line-height:1; }}
.stat-label {{ font-size:0.7rem; color:rgba(255,255,255,0.55) !important; text-transform:uppercase; letter-spacing:1px; }}

.form-card {{ background:rgba(255,255,255,0.97) !important; border-radius:18px; padding:1.8rem; box-shadow:0 8px 32px rgba(0,0,0,0.25); margin-bottom:1rem; border-left:5px solid #2980b9; }}
.form-card-title {{ font-size:1.1rem; font-weight:700; color:#1a3c5e !important; margin-bottom:1rem; padding-bottom:0.5rem; border-bottom:2px solid #f0f4f8; }}

.stSelectbox label, .stNumberInput label, .stTextInput label {{
    color:white !important; font-weight:700 !important; font-size:0.9rem !important;
    background:linear-gradient(135deg,#1a3c5e,#2980b9) !important;
    padding:3px 10px !important; border-radius:20px !important; display:inline-block !important; margin-bottom:4px !important;
}}

.stButton>button {{
    background:linear-gradient(135deg,#0d47a1,#1565c0,#0097a7);
    color:white !important; border-radius:15px; padding:0.9rem 2rem;
    font-size:1.1rem; font-weight:700; width:100%; border:none;
    box-shadow:0 5px 20px rgba(13,71,161,0.4);
    font-family:'Rajdhani',sans-serif; letter-spacing:1px;
}}

.result-card {{ border-radius:20px; padding:2rem; text-align:center; margin:1rem 0; box-shadow:0 8px 30px rgba(0,0,0,0.3); animation:fadeInUp 0.6s ease; }}
@keyframes fadeInUp {{ from{{opacity:0;transform:translateY(20px)}} to{{opacity:1;transform:translateY(0)}} }}

.reco-card {{ background:rgba(255,255,255,0.96); border-radius:18px; padding:1.5rem; margin:1rem 0; box-shadow:0 6px 25px rgba(0,0,0,0.2); }}
.reco-conseil {{ padding:8px 12px; margin:6px 0; border-radius:10px; background:#f8f9fa; border-left:4px solid #2980b9; color:#1a3c5e !important; font-size:0.95rem; }}

.hist-item {{ background:rgba(255,255,255,0.1); border-radius:14px; padding:14px 16px; margin:8px 0; border:1px solid rgba(255,255,255,0.08); transition:all 0.2s; }}
.hist-item:hover {{ background:rgba(255,255,255,0.16); transform:translateX(3px); }}

.apropos-card {{ background:rgba(255,255,255,0.96); border-radius:18px; padding:2rem; box-shadow:0 8px 32px rgba(0,0,0,0.2); }}
.apropos-card h3,.apropos-card p,.apropos-card li,.apropos-card b {{ color:#1a3c5e !important; }}

.stMarkdown p, h1, h2, h3 {{ color:white !important; }}

@keyframes pulse {{ 0%{{box-shadow:0 0 0 0 rgba(231,76,60,0.7)}} 70%{{box-shadow:0 0 0 10px rgba(231,76,60,0)}} 100%{{box-shadow:0 0 0 0 rgba(231,76,60,0)}} }}
.urgent {{ animation:pulse 2s infinite; }}
</style>
""", unsafe_allow_html=True)

# SLIDESHOW
st.markdown('<div class="slideshow"><div class="slide"></div><div class="slide"></div><div class="slide"></div><div class="slide"></div></div>', unsafe_allow_html=True)

# HERO
st.markdown(f"""
<div class="hero-section">
    {'<img src="data:image/png;base64,' + logo_b64 + '" style="width:90px;margin-bottom:0.5rem;">' if logo_b64 else '🏥'}
    <div class="hero-title">OBESITY RISK AI</div>
    <div class="hero-subtitle">Système intelligent d'aide à la décision médicale</div>
    <div class="hero-badge">🎓 Centrale Casablanca — Coding Week 2026</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        {'<img src="data:image/png;base64,' + logo_b64 + '" style="width:60px;">' if logo_b64 else ''}
        <div style="font-family:Rajdhani,sans-serif;font-size:1.4rem;font-weight:700;
                    letter-spacing:2px;color:#29b6f6 !important;
                    text-shadow:0 0 15px rgba(41,182,246,0.5);">OBESITY RISK</div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.45) !important;
                    text-transform:uppercase;letter-spacing:2px;">Medical AI Platform</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.connecte:
        nb = len(st.session_state.historique)
        st.markdown(f"""
        <div class="medecin-badge">
            <div style="font-size:1.5rem">👨‍⚕️</div>
            <div style="font-weight:700;font-size:0.95rem;">{st.session_state.medecin_nom}</div>
            <div style="font-size:0.72rem;color:rgba(255,255,255,0.55) !important;">{st.session_state.medecin_email}</div>
        </div>
        <div style="display:flex;gap:8px;margin:8px 0;">
            <div class="stat-box" style="flex:1"><div class="stat-number">{nb}</div><div class="stat-label">Analyses</div></div>
            <div class="stat-box" style="flex:1"><div class="stat-number">7</div><div class="stat-label">Classes</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    pages_nav = [
        ("🏠", "Accueil",         "🏠 Accueil"),
        ("👤", "Analyse Patient", "👤 Analyse Patient"),
        ("🧑", "Espace Patient",  "🧑 Espace Patient"),
        ("📋", "Historique",      "📋 Historique"),
        ("📊", "Statistiques",    "📊 Statistiques"),
        ("👥", "Notre Équipe",    "👥 Notre Équipe"),
        ("ℹ️", "À propos",       "ℹ️ À propos"),
    ]
    for icon, label, key in pages_nav:
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            if key in ["👤 Analyse Patient", "📋 Historique"] and not st.session_state.connecte:
                st.warning("🔒 Connexion requise")
            else:
                st.session_state.page = key
                st.rerun()

    st.markdown("---")
    if st.session_state.connecte:
        if st.button("🚪 Se déconnecter", use_container_width=True):
            st.session_state.connecte = False
            st.session_state.medecin_nom = ""
            st.session_state.medecin_email = ""
            st.session_state.page = "🏠 Accueil"
            st.rerun()
    else:
        if st.button("🔐 Se connecter", use_container_width=True):
            st.session_state.page = "🔐 Connexion"
            st.rerun()

    couleur_s = "#2ecc71" if st.session_state.connecte else "#e74c3c"
    status_t  = "Connecté" if st.session_state.connecte else "Non connecté"
    st.markdown(f"""
    <div style="margin-top:10px;text-align:center;">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                     background:{couleur_s};margin-right:6px;"></span>
        <span style="font-size:0.76rem;color:rgba(255,255,255,0.45) !important;">{status_t}</span>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ============================================
# PAGE CONNEXION
# ============================================
if page == "🔐 Connexion":
    st.markdown("<h2 style='text-align:center'>🔐 Connexion Médecin</h2>", unsafe_allow_html=True)
    _, col_form, _ = st.columns([1, 2, 1])
    with col_form:
        st.markdown('<div style="background:rgba(255,255,255,0.97);border-radius:20px;padding:2rem;box-shadow:0 10px 40px rgba(0,0,0,0.3);">', unsafe_allow_html=True)

        # Onglets Connexion / Inscription
        tab_conn, tab_inscr = st.tabs(["🔐 Connexion", "📝 Inscription"])

        with tab_conn:
            email    = st.text_input("📧 Email professionnel", placeholder="dr.nom@hospital.ma", key="login_email")
            password = st.text_input("🔑 Mot de passe", type="password", key="login_pass")
            if st.button("🚀 Se connecter", use_container_width=True, key="btn_login"):
                if email in MEDECINS and MEDECINS[email]["password"] == password:
                    st.session_state.connecte      = True
                    st.session_state.medecin_nom   = MEDECINS[email]["nom"]
                    st.session_state.medecin_email = email
                    st.session_state.page          = "🏠 Accueil"
                    st.success(f"✅ Bienvenue, {MEDECINS[email]['nom']} !")
                    st.rerun()
                else:
                    st.error("❌ Email ou mot de passe incorrect")
            st.markdown("""
            <p style="color:#888;font-size:0.82rem;text-align:center;margin-top:10px;">
            Pas encore de compte ? Cliquez sur <b>Inscription</b> ci-dessus.
            </p>""", unsafe_allow_html=True)

        with tab_inscr:
            new_nom   = st.text_input("👨‍⚕️ Nom complet", placeholder="Dr. Prénom Nom", key="reg_nom")
            new_email = st.text_input("📧 Email professionnel", placeholder="dr.nom@hospital.ma", key="reg_email")
            new_pass  = st.text_input("🔑 Mot de passe", type="password", key="reg_pass")
            new_pass2 = st.text_input("🔑 Confirmer le mot de passe", type="password", key="reg_pass2")
            if st.button("✅ Créer mon compte", use_container_width=True, key="btn_register"):
                if not new_nom or not new_email or not new_pass:
                    st.error("❌ Veuillez remplir tous les champs")
                elif new_pass != new_pass2:
                    st.error("❌ Les mots de passe ne correspondent pas")
                elif new_email in MEDECINS:
                    st.error("❌ Cet email est déjà utilisé")
                elif len(new_pass) < 6:
                    st.error("❌ Le mot de passe doit contenir au moins 6 caractères")
                else:
                    MEDECINS[new_email] = {"password": new_pass, "nom": new_nom}
                    st.success(f"✅ Compte créé pour **{new_nom}** ! Connectez-vous dans l'onglet Connexion.")
            st.markdown("""
            <p style="color:#888;font-size:0.82rem;text-align:center;margin-top:10px;">
            Déjà un compte ? Cliquez sur <b>Connexion</b> ci-dessus.
            </p>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# PAGE ACCUEIL
# ============================================
elif page == "🏠 Accueil":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("## 👋 Bienvenue sur Obesity Risk AI !")
        st.markdown("**Obesity Risk AI** aide les médecins à estimer le **risque d'obésité** avec explications SHAP et recommandations personnalisées.")
        st.markdown("### 🔄 Comment ça marche ?")
        for icon, titre, desc in [
            ("1️⃣","Connexion sécurisée","Le médecin s'authentifie"),
            ("2️⃣","Saisie des données","16 variables du patient"),
            ("3️⃣","Analyse par IA","Prédiction LightGBM en temps réel"),
            ("4️⃣","Explication SHAP","Facteurs déterminants"),
            ("5️⃣","Recommandations","Conseils médicaux personnalisés"),
            ("6️⃣","Historique","Suivi des analyses passées"),
        ]:
            st.markdown(f'<div style="background:rgba(255,255,255,0.12);border-radius:10px;padding:0.8rem 1rem;margin:0.3rem 0;border-left:4px solid #29b6f6;"><b style="color:white">{icon} {titre}</b><br><span style="color:rgba(255,255,255,0.75);font-size:0.88rem;">{desc}</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown("### 🎯 Les 7 niveaux")
        for emoji, niveau, color in [
            ("🔵","Insufficient Weight","#3498db"),("🟢","Normal Weight","#2ecc71"),
            ("🟡","Overweight Level I","#f1c40f"),("🟠","Overweight Level II","#e67e22"),
            ("🔴","Obesity Type I","#e74c3c"),("🔴","Obesity Type II","#c0392b"),
            ("🔴","Obesity Type III","#922b21"),
        ]:
            st.markdown(f'<div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:0.5rem 1rem;margin:0.3rem 0;color:white;border-left:3px solid {color};">{emoji} {niveau}</div>', unsafe_allow_html=True)
        if not st.session_state.connecte:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔐 Se connecter", use_container_width=True):
                st.session_state.page = "🔐 Connexion"; st.rerun()

# ============================================
# PAGE ANALYSE PATIENT
# ============================================
elif page == "👤 Analyse Patient":
    if not st.session_state.connecte:
        st.warning("🔒 Connexion requise.")
        if st.button("🔐 Se connecter"):
            st.session_state.page = "🔐 Connexion"; st.rerun()
    else:
        st.markdown(f'<div style="background:linear-gradient(135deg,#0d2137,#1565c0);border-radius:15px;padding:1rem 1.5rem;margin-bottom:1rem;"><h2 style="color:white;margin:0;">👤 Analyse Patient — {st.session_state.medecin_nom}</h2></div>', unsafe_allow_html=True)
        nom_patient = st.text_input("🧑 Nom du patient (optionnel)", placeholder="Ex: Mohamed Alaoui")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="form-card"><div class="form-card-title">👤 Informations physiques</div>', unsafe_allow_html=True)
            genre  = st.selectbox("Genre", ["Male","Female"])
            age    = st.number_input("Âge", min_value=10, max_value=80, value=25)
            taille = st.number_input("Taille (m)", min_value=1.40, max_value=2.10, value=1.70, step=0.01, format="%.2f")
            poids  = st.number_input("Poids (kg)", min_value=30, max_value=200, value=70)
            imc = poids / (taille**2)
            if imc < 18.5:  st.info(f"📊 IMC : **{imc:.1f}** — Poids insuffisant")
            elif imc < 25:  st.success(f"📊 IMC : **{imc:.1f}** — Poids normal ✅")
            elif imc < 30:  st.warning(f"📊 IMC : **{imc:.1f}** — Surpoids ⚠️")
            else:           st.error(f"📊 IMC : **{imc:.1f}** — Obésité 🔴")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="form-card"><div class="form-card-title">🍔 Habitudes alimentaires</div>', unsafe_allow_html=True)
            antecedents = st.selectbox("Antécédents familiaux d'obésité", ["yes","no"])
            favc  = st.selectbox("Fast-food fréquent", ["yes","no"])
            fcvc  = st.number_input("Fréquence légumes (1-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1, format="%.1f")
            ncp   = st.number_input("Repas par jour", min_value=1.0, max_value=4.0, value=3.0, step=0.5, format="%.1f")
            caec  = st.selectbox("Grignotage entre repas", ["no","Sometimes","Frequently","Always"])
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="form-card"><div class="form-card-title">🏃 Activité & Mode de vie</div>', unsafe_allow_html=True)
            faf    = st.number_input("Sport (jours/semaine)", min_value=0.0, max_value=3.0, value=1.0, step=0.5, format="%.1f")
            tue    = st.number_input("Temps écran (h/jour)", min_value=0.0, max_value=2.0, value=1.0, step=0.25, format="%.2f")
            ch2o   = st.number_input("Eau (litres/jour)", min_value=1.0, max_value=3.0, value=2.0, step=0.1, format="%.1f")
            mtrans = st.selectbox("Mode de transport", ["Public_Transportation","Walking","Automobile","Motorbike","Bike"])
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="form-card"><div class="form-card-title">🚬 Autres habitudes</div>', unsafe_allow_html=True)
            smoke = st.selectbox("Fumeur", ["no","yes"])
            calc  = st.selectbox("Consommation d'alcool", ["no","Sometimes","Frequently","Always"])
            scc   = st.selectbox("Surveille ses calories", ["no","yes"])
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Lancer l'analyse IA", use_container_width=True):
            donnees = {
                'Gender':genre,'Age':float(age),'Height':float(taille),'Weight':float(poids),
                'family_history_with_overweight':antecedents,'FAVC':favc,'FCVC':float(fcvc),
                'NCP':float(ncp),'CAEC':caec,'SMOKE':smoke,'CH2O':float(ch2o),
                'SCC':scc,'FAF':float(faf),'TUE':float(tue),'CALC':calc,'MTRANS':mtrans
            }
            df_patient = pd.DataFrame([donnees])
            for col_name, le in encodeurs.items():
                if col_name in df_patient.columns:
                    df_patient[col_name] = le.transform(df_patient[col_name].astype(str))
            cols_num = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']
            cols_cat = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']
            df_num_sc = pd.DataFrame(scaler.transform(df_patient[cols_num]), columns=cols_num)
            df_sc = pd.concat([df_patient[cols_cat].astype(float), df_num_sc], axis=1)
            colonnes = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
            df_sc = df_sc[colonnes]
            prediction = modele.predict(df_sc)[0]
            probas     = modele.predict_proba(df_sc)[0]
            niveaux = {0:("🔵","Insufficient Weight","#3498db"),1:("🟢","Normal Weight","#2ecc71"),
                       2:("🟡","Overweight Level I","#f1c40f"),3:("🟠","Overweight Level II","#e67e22"),
                       4:("🔴","Obesity Type I","#e74c3c"),5:("🔴","Obesity Type II","#c0392b"),
                       6:("🔴","Obesity Type III","#922b21")}
            emoji, label, couleur = niveaux[prediction]

            # Historique
            st.session_state.historique.insert(0, {
                "date":      datetime.now().strftime("%d/%m/%Y %H:%M"),
                "patient":   nom_patient if nom_patient else "Patient anonyme",
                "medecin":   st.session_state.medecin_nom,
                "prediction": label, "emoji": emoji, "couleur": couleur,
                "confiance": f"{probas[prediction]*100:.1f}%", "imc": f"{imc:.1f}",
            })

            st.markdown("---")
            st.markdown("### 🎯 Résultat")
            urgent_class = "urgent" if prediction >= 4 else ""
            st.markdown(f"""
            <div class="result-card {urgent_class}" style="background:{couleur}33;border:3px solid {couleur};">
                <div style="font-size:3.5rem">{emoji}</div>
                <div style="font-family:Rajdhani,sans-serif;font-size:2rem;font-weight:700;color:white;">{label}</div>
                <div style="color:rgba(255,255,255,0.85);font-size:1.1rem;">
                    Confiance : <b>{probas[prediction]*100:.1f}%</b> &nbsp;|&nbsp; IMC : <b>{imc:.1f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Probabilités")
            for idx, (em, lbl, col) in niveaux.items():
                st.progress(float(probas[idx]), text=f"{em} {lbl} : {probas[idx]*100:.1f}%")

            # RECOMMANDATIONS
            st.markdown("---")
            st.markdown("### 💊 Recommandations Médicales")
            reco = RECOMMANDATIONS[prediction]
            uc   = "#e74c3c" if prediction >= 4 else "#2ecc71"
            st.markdown(f'<div class="reco-card"><h3 style="color:{reco["couleur"]} !important;">{reco["titre"]}</h3><div style="background:{uc}22;border-radius:8px;padding:8px 14px;margin-bottom:12px;border-left:4px solid {uc};"><b style="color:{uc} !important;">⏱️ {reco["urgence"]}</b></div>', unsafe_allow_html=True)
            for conseil in reco["conseils"]:
                st.markdown(f'<div class="reco-conseil">{conseil}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # SHAP
            st.markdown("---")
            st.markdown("### 🔬 Explication SHAP")
            try:
                explainer   = shap.TreeExplainer(modele)
                shap_values = explainer.shap_values(df_sc)
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.summary_plot(shap_values, df_sc, plot_type="bar", show=False)
                plt.title("Impact des variables sur la prédiction", color='white')
                fig.patch.set_alpha(0.0); ax.set_facecolor('none')
                st.pyplot(fig); plt.close()
            except Exception as e:
                st.warning(f"⚠️ SHAP non disponible : {e}")

# ============================================
# PAGE ESPACE PATIENT
# ============================================
elif page == "🧑 Espace Patient":
    st.markdown(f'<div style="background:linear-gradient(135deg,#0d3b2e,#1a7a4a);border-radius:15px;padding:1rem 1.5rem;margin-bottom:1rem;"><h2 style="color:white;margin:0;">🧑 Espace Patient — Comprendre mon résultat</h2></div>', unsafe_allow_html=True)

    st.markdown("### 📋 Entrez votre résultat de prédiction")
    niveau_choisi = st.selectbox("Mon niveau d'obésité prédit :", [
        "0 — Insufficient Weight",
        "1 — Normal Weight",
        "2 — Overweight Level I",
        "3 — Overweight Level II",
        "4 — Obesity Type I",
        "5 — Obesity Type II",
        "6 — Obesity Type III",
    ])
    idx_niveau = int(niveau_choisi.split(" — ")[0])
    reco = RECOMMANDATIONS[idx_niveau]
    uc   = "#e74c3c" if idx_niveau >= 4 else "#2ecc71"

    st.markdown("---")
    st.markdown("### 💊 Vos Recommandations Personnalisées")
    st.markdown(f'<div class="reco-card"><h3 style="color:{reco["couleur"]} !important;">{reco["titre"]}</h3><div style="background:{uc}22;border-radius:8px;padding:8px 14px;margin-bottom:12px;border-left:4px solid {uc};"><b style="color:{uc} !important;">⏱️ {reco["urgence"]}</b></div>', unsafe_allow_html=True)
    for conseil in reco["conseils"]:
        st.markdown(f'<div class="reco-conseil">{conseil}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ Comprendre les niveaux d'obésité")
    for emoji, niveau, color, desc in [
        ("🔵","Insufficient Weight","#3498db","IMC < 18.5 — Poids insuffisant, risque de carences"),
        ("🟢","Normal Weight","#2ecc71","IMC 18.5–24.9 — Poids idéal, continuez ainsi !"),
        ("🟡","Overweight Level I","#f1c40f","IMC 25–27.4 — Léger surpoids, quelques ajustements"),
        ("🟠","Overweight Level II","#e67e22","IMC 27.5–29.9 — Surpoids modéré, action recommandée"),
        ("🔴","Obesity Type I","#e74c3c","IMC 30–34.9 — Obésité légère, suivi médical requis"),
        ("🔴","Obesity Type II","#c0392b","IMC 35–39.9 — Obésité modérée, prise en charge urgente"),
        ("🔴","Obesity Type III","#922b21","IMC ≥ 40 — Obésité sévère, intervention immédiate"),
    ]:
        st.markdown(f'<div style="background:rgba(255,255,255,0.1);border-radius:10px;padding:0.7rem 1rem;margin:0.3rem 0;border-left:4px solid {color};"><b style="color:white;">{emoji} {niveau}</b><br><span style="color:rgba(255,255,255,0.7);font-size:0.88rem;">{desc}</span></div>', unsafe_allow_html=True)

# ============================================
# PAGE HISTORIQUE 
# ============================================
elif page == "📋 Historique":
    if not st.session_state.connecte:
        st.warning("🔒 Connexion requise.")
    else:
        st.markdown("## 📋 Historique des Analyses")
        st.markdown(f"**{st.session_state.medecin_nom}** — {len(st.session_state.historique)} analyse(s)")
        if not st.session_state.historique:
            st.info("Aucune analyse effectuée pour l'instant.")
        else:
            col_exp, col_v = st.columns([4,1])
            with col_exp:
                df_export = pd.DataFrame(st.session_state.historique)
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Exporter en CSV",
                    data=csv,
                    file_name=f"historique_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
            with col_v:
                if st.button("🗑️ Vider"):
                    st.session_state.historique = []; st.rerun()
            for entree in st.session_state.historique:
                st.markdown(f"""
                <div class="hist-item">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div><span style="font-size:1.4rem">{entree['emoji']}</span>
                             <b style="color:white;margin-left:8px;">{entree['patient']}</b></div>
                        <div style="text-align:right;">
                            <div style="color:{entree['couleur']} !important;font-weight:700;">{entree['prediction']}</div>
                            <div style="font-size:0.78rem;color:rgba(255,255,255,0.45) !important;">{entree['date']}</div>
                        </div>
                    </div>
                    <div style="margin-top:5px;font-size:0.8rem;color:rgba(255,255,255,0.55) !important;">
                        👨‍⚕️ {entree['medecin']} &nbsp;|&nbsp; Confiance : {entree['confiance']} &nbsp;|&nbsp; IMC : {entree['imc']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### 📊 Résumé")
            df_hist = pd.DataFrame(st.session_state.historique)
            counts  = df_hist['prediction'].value_counts()
            fig, ax = plt.subplots(figsize=(8,3))
            counts.plot(kind='barh', ax=ax, color='#29b6f6')
            ax.set_facecolor('none'); fig.patch.set_alpha(0.0)
            ax.tick_params(colors='white'); plt.tight_layout()
            st.pyplot(fig); plt.close()

# ============================================
# PAGE STATISTIQUES
# ============================================
elif page == "📊 Statistiques":
    st.markdown("## 📊 Statistiques du Dataset")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=544)
        X = dataset.data.features; y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Patients", df.shape[0])
        col2.metric("📊 Variables", df.shape[1])
        col3.metric("🎯 Classes", y.nunique())
        fig, ax = plt.subplots(figsize=(10,4))
        y.value_counts().plot(kind='bar', ax=ax, color='#29b6f6')
        ax.set_xlabel("Niveau d'obésité", color='white'); ax.set_ylabel("Patients", color='white')
        ax.tick_params(colors='white'); plt.xticks(rotation=45, color='white')
        fig.patch.set_alpha(0.0); ax.set_facecolor('none')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        col1, col2 = st.columns(2)
        for col_p, col_d, color, title in [(col1,'Age','#2ecc71',"Âge"),(col2,'Weight','#e74c3c',"Poids (kg)")]:
            with col_p:
                fig, ax = plt.subplots()
                df[col_d].hist(ax=ax, bins=10, color=color)
                ax.set_title(f"Distribution — {title}", color='white')
                ax.tick_params(colors='white'); fig.patch.set_alpha(0.0); ax.set_facecolor('none')
                st.pyplot(fig); plt.close()
    except Exception as e:
        st.warning(f"⚠️ Dataset non disponible : {e}")

# ============================================
# PAGE Notre ÉQUIPE
# ============================================
elif page == "👥 Notre Équipe":
    st.markdown("## 👥 Notre Équipe — TEAM DATA HEALERS")
    col_img, col_noms = st.columns([2,1])
    with col_img:
        team_path = os.path.join(APP_DIR, 'team.jpg')
        if os.path.exists(team_path): st.image(team_path, caption="Coding Week 2026")
        else: st.info("📸 Ajoutez app/team.jpg")
    with col_noms:
        for nom, color in [("Meryem Querchi","#29b6f6"),("Amina Boutalmaouine","#2ecc71"),
                           ("Douae Amghar","#e74c3c"),("Hajar Azoud","#f39c12"),("Hajar Dyaz","#9b59b6")]:
            st.markdown(f'<div style="background:rgba(255,255,255,0.12);border-radius:10px;padding:0.8rem 1rem;margin:0.4rem 0;border-left:5px solid {color};"><span style="font-weight:700;color:white;">👩‍💻 {nom}</span></div>', unsafe_allow_html=True)

# ============================================
# PAGE À PROPOS
# ============================================
elif page == "ℹ️ À propos":
    st.markdown("## ℹ️ À propos du projet")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="apropos-card"><h3>🎯 Objectif</h3><p>Outil clinique ML pour estimer le risque d\'obésité avec explications SHAP et recommandations personnalisées.</p><h3>📁 Dataset UCI #544</h3><ul><li><b>2111 patients</b> — 17 variables — 7 classes</li></ul></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="apropos-card"><h3>🛠️ Technologies</h3><ul><li><b>LightGBM</b> — ~96% précision</li><li><b>SHAP</b> — Explicabilité</li><li><b>Streamlit</b> — Interface web</li><li><b>GitHub Actions</b> — CI/CD</li></ul><h3>✨ Fonctionnalités</h3><ul><li>🔐 Auth médecin</li><li>📋 Historique</li><li>💊 Recommandations</li><li>🔬 SHAP</li></ul></div>', unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown('<div style="text-align:center;color:rgba(255,255,255,0.35);font-size:0.76rem;">🏥 Obesity Risk AI &nbsp;|&nbsp; Centrale Casablanca — Coding Week 2026 &nbsp;|&nbsp; Explainable ML with SHAP</div>', unsafe_allow_html=True)