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
Raymonda has experience in Talent Acquisition and analytics, working with stakeholders and recruitment pipelines.
Strong communication, reporting, data analysis, and Excel. Familiar with SQL basics and Python for automation.
Worked on AI screening prototype and dashboard reporting.
"""

SAMPLE_JD = """
We are hiring a Talent Intelligence Analyst.
Must have: data analysis, reporting, stakeholder management, Excel, communication.
Nice to have: Python, SQL, machine learning, dashboarding, cloud.
Responsibilities include analyzing recruiting data, building dashboards, and recommending hiring decisions.
"""

SAMPLE_CANDIDATES = [
    {
        "Candidate": "Candidate A (You)",
        "Resume": SAMPLE_RESUME
    },
    {
        "Candidate": "Candidate B",
        "Resume": """
        Data analyst with strong SQL, Python and dashboard skills. Built machine learning models for classification.
        Experience with reporting and stakeholder presentations. Worked on cloud deployments (AWS).
        """
    },
    {
        "Candidate": "Candidate C",
        "Resume": """
        HR generalist with strong communication and stakeholder management. Basic Excel and reporting.
        Limited exposure to data analysis tools; no coding experience.
        """
    }
]

# ------------------ HELPERS ------------------
SKILLS_DB = [
    "python", "java", "sql", "machine learning", "data analysis", "analytics",
    "cloud", "aws", "communication", "reporting", "dashboard", "stakeholder",
    "excel", "statistics", "power bi", "tableau"
]

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_skills(text: str) -> list:
    t = clean_text(text)
    found = []
    for s in SKILLS_DB:
        # simple phrase search
        if s in t:
            found.append(s)
    return sorted(list(set(found)))

def ai_similarity_score(resume: str, jd: str):
    """
    TF-IDF on [resume, jd] then cosine similarity.
    Returns score (0-100) and optional debug info.
    """
    resume_c = clean_text(resume)
    jd_c = clean_text(jd)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume_c, jd_c])

    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]  # 0..1
    score = int(round(sim * 100))

    debug = {
        "vocab_size": len(vectorizer.get_feature_names_out()),
        "similarity_raw": sim
    }
    return score, debug

def label_from_score(score: int) -> str:
    if score >= 80:
        return "Excellent Match 🔥"
    if score >= 60:
        return "Strong Match ✅"
    if score >= 40:
        return "Moderate Match 👍"
    return "Weak Match ⚠️"

def recommendation_from_score(score: int) -> str:
    if score >= 80:
        return "Recommend shortlist"
    if score >= 60:
        return "Recommend next round"
    if score >= 40:
        return "Consider if role is flexible"
    return "Not recommended (gap too large)"

# ------------------ SESSION STATE ------------------
if "resume_text" not in st.session_state:
    st.session_state.resume_text = SAMPLE_RESUME if use_samples else ""
if "jd_text" not in st.session_state:
    st.session_state.jd_text = SAMPLE_JD if use_samples else ""
if "last_score" not in st.session_state:
    st.session_state.last_score = None
if "last_debug" not in st.session_state:
    st.session_state.last_debug = None
if "last_matched" not in st.session_state:
    st.session_state.last_matched = []
if "last_missing" not in st.session_state:
    st.session_state.last_missing = []

# ------------------ PAGE 1: INPUT ------------------
if mode == "1) Input":
    colA, colB = st.columns(2)

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📄 Resume Input (Candidate)")
        st.session_state.resume_text = st.text_area("Paste resume text here", value=st.session_state.resume_text, height=260)
        st.markdown('<div class="small">Tip: Use the sample toggle in the sidebar for an instant demo.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Job Description Input (JD)")
        st.session_state.jd_text = st.text_area("Paste job description here", value=st.session_state.jd_text, height=260)
        st.markdown('<div class="small">POC goal: show how AI compares text content and produces a score + explanation.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("🚀 Run TalentPulse AI Matching"):
        if not st.session_state.resume_text.strip() or not st.session_state.jd_text.strip():
            st.error("Please fill in both Resume and Job Description.")
        else:
            with st.spinner("Running TF-IDF model + cosine similarity..."):
                score, debug = ai_similarity_score(st.session_state.resume_text, st.session_state.jd_text)

            resume_sk = extract_skills(st.session_state.resume_text)
            jd_sk = extract_skills(st.session_state.jd_text)

            matched = sorted(list(set(resume_sk) & set(jd_sk)))
            missing = sorted(list(set(jd_sk) - set(resume_sk)))

            st.session_state.last_score = score
            st.session_state.last_debug = debug
            st.session_state.last_matched = matched
            st.session_state.last_missing = missing

            st.success("Done! Go to '2) Matching Results' in the sidebar.")

# ------------------ PAGE 2: MATCHING RESULTS ------------------
if mode == "2) Matching Results":
    if st.session_state.last_score is None:
        st.warning("Run the matching first. Go to '1) Input' and click 'Run TalentPulse AI Matching'.")
    else:
        score = st.session_state.last_score
        label = label_from_score(score)

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🎯 AI Match Score")
            st.progress(score / 100)
            st.metric("Final Score", f"{score}%")
            st.success(label)
            st.markdown(f"<div class='small'><b>Recommendation:</b> {recommendation_from_score(score)}</div>", unsafe_allow_html=True)

            if show_debug and st.session_state.last_debug:
                d = st.session_state.last_debug
                st.markdown("<hr>", unsafe_allow_html=True)
                st.caption("Debug (AI details)")
                st.write({
                    "TF-IDF vocabulary size": d["vocab_size"],
                    "Raw cosine similarity (0..1)": round(d["similarity_raw"], 4)
                })
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("✅ Skill Alignment")
            st.caption("Matched Skills")
            if st.session_state.last_matched:
                for s in st.session_state.last_matched:
                    st.markdown(f"<span class='badge badge-ok'>{s}</span>", unsafe_allow_html=True)
            else:
                st.info("No matched skills detected based on the skill list.")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.caption("Missing Skills (Skill Gaps)")
            if st.session_state.last_missing:
                for s in st.session_state.last_missing:
                    st.markdown(f"<span class='badge badge-miss'>{s}</span>", unsafe_allow_html=True)
            else:
                st.success("No missing skills detected. Nice.")
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PAGE 3: RANKING ------------------
if mode == "3) Ranking":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Candidate Ranking (POC Simulation)")
    st.caption("This ranks multiple candidates against the same Job Description using TF-IDF similarity.")

    if not st.session_state.jd_text.strip():
        st.error("Please provide a Job Description in '1) Input' first.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        rows = []
        for c in SAMPLE_CANDIDATES:
            score, _ = ai_similarity_score(c["Resume"], st.session_state.jd_text)
            rows.append({
                "Candidate": c["Candidate"],
                "Match Score (%)": score,
                "Match Level": label_from_score(score),
                "Recommendation": recommendation_from_score(score)
            })

        df = pd.DataFrame(rows).sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption("Tip: Change your JD in the Input tab and see the ranking update — that’s your POC verification.")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PAGE 4: EXPLANATION / XAI ------------------
if mode == "4) Explanation (XAI)":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Explainable AI (XAI) + POC Verification")

    st.markdown("""
**What the POC demonstrates (CLO2):**
- An interactive interface where users provide inputs (Resume + Job Description).
- An AI-based text matching method to solve a real-world problem (resume screening).
- Verification of the concept using measurable output (match score + ranking + skill gaps).

**AI Method Used: TF-IDF + Cosine Similarity**
1. Convert resume and job description into TF-IDF vectors (weighted keyword importance).
2. Compute cosine similarity between the two vectors (0 to 1).
3. Convert similarity into a score: `Score = similarity × 100`.

**Skill Gap Analysis**
- A curated skill list is checked in both texts.
- The app shows:
  - Matched skills (alignment)
  - Missing skills (gaps)
""")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("✅ How this verifies the concept")
    st.markdown("""
Try this to prove the POC works:
- Change the job description (add/remove “python”, “sql”, “machine learning”).
- Re-run the matching and observe:
  - The score changes.
  - The matched/missing skill list changes.
  - Candidate ranking changes.

That behavior is the “proof” that the AI concept is functioning.
""")
    st.markdown('</div>', unsafe_allow_html=True)
