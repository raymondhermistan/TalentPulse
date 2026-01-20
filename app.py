# TalentPulse POC (CLO2) - Streamlit Prototype
# AI method: TF-IDF + Cosine Similarity (plus skill-gap extraction)
# Deployed-ready for Streamlit Community Cloud

import streamlit as st
import pandas as pd
import re
from math import floor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="TalentPulse", page_icon="🤖", layout="wide")

# ------------------ SIMPLE THEME / STYLE ------------------
st.markdown("""
<style>
.big-title { font-size:42px; font-weight:800; color:#1f77ff; margin-bottom:0px;}
.sub { color:#94a3b8; margin-top:0px; }
.card { background:#0b1220; border:1px solid #1f2a44; padding:18px; border-radius:14px; }
.badge { display:inline-block; padding:6px 12px; border-radius:999px; font-size:13px; margin:4px 6px 0 0; color:white; background:#2563eb; }
.badge-miss { background:#ef4444; }
.badge-ok { background:#22c55e; }
.small { font-size:13px; color:#cbd5e1; }
hr { border: none; height: 1px; background: #1f2a44; margin: 12px 0 18px 0;}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="big-title">🤖 TalentPulse</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Proof of Concept (POC): AI Resume ↔ Job Description Matching using TF-IDF + Cosine Similarity</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ POC Controls")

mode = st.sidebar.radio("Navigation", ["1) Input", "2) Matching Results", "3) Ranking", "4) Explanation (XAI)"])

use_samples = st.sidebar.checkbox("Use sample data (recommended for demo)", value=True)
show_debug = st.sidebar.checkbox("Show AI debug details (optional)", value=False)

# ------------------ SAMPLE DATA ------------------
SAMPLE_RESUME = """
Raymonda


