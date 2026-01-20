import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional readers (only used if installed)
try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


st.set_page_config(page_title="TalentPulse", page_icon="🤖", layout="wide")

st.title("🤖 TalentPulse")
st.caption("POC: Resume ↔ JD Matching (TF-IDF + Cosine Similarity) with Upload + Name Detection")
st.markdown("---")

# -------------------- Helpers --------------------
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

def ai_similarity_score(resume: str, jd: str) -> int:
    resume_c = clean_text(resume)
    jd_c = clean_text(jd)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume_c, jd_c])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]  # 0..1
    return int(round(sim * 100))

def extract_skills(text: str) -> list:
    t = clean_text(text)
    found = []
    for s in SKILLS_DB:
        if s in t:
            found.append(s)
    return sorted(list(set(found)))

def read_txt(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8", errors="ignore")

def read_docx(uploaded_file) -> str:
    if docx is None:
        return ""
    d = docx.Document(uploaded_file)
    return "\n".join([p.text for p in d.paragraphs])

def read_pdf(uploaded_file) -> str:
    if PdfReader is None:
        return ""
    reader = PdfReader(uploaded_file)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

def read_any_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return read_txt(uploaded_file)
    if name.endswith(".docx"):
        return read_docx(uploaded_file)
    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    return ""

def guess_candidate_name(resume_text: str) -> str:
    """
    Best-effort name detection:
    - Looks at the first ~8 lines
    - Picks a likely 'Name' line (2-4 words, letters only)
    - Avoids common CV headers like "resume", "curriculum vitae"
    """
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    top = lines[:8]

    blacklist = {"resume", "curriculum vitae", "cv", "profile", "contact", "summary"}
    for ln in top:
        low = ln.lower()
        if any(b in low for b in blacklist):
            continue

        # strip email/phone if same line
        ln2 = re.sub(r"\S+@\S+", "", ln)
        ln2 = re.sub(r"(\+?\d[\d\s\-\(\)]{7,})", "", ln2).strip()

        # likely name: 2-4 words, letters and spaces only
        if re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+){1,3}", ln2):
            return ln2

    return "Candidate"

def label_from_score(score: int) -> str:
    if score >= 80:
        return "Excellent Match 🔥"
    if score >= 60:
        return "Strong Match ✅"
    if score >= 40:
        return "Moderate Match 👍"
    return "Weak Match ⚠️"

# -------------------- UI --------------------
st.subheader("📤 Upload Files")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Resumes (upload 1 or more)")
    resumes = st.file_uploader(
        "Upload resume files (TXT / DOCX / PDF)",
        type=["txt", "docx", "pdf"],
        accept_multiple_files=True
    )

with col2:
    st.markdown("### Job Description (upload 1)")
    jd_file = st.file_uploader(
        "Upload JD file (TXT / DOCX / PDF)",
        type=["txt", "docx", "pdf"],
        accept_multiple_files=False
    )

st.markdown("---")

# Read JD
jd_text = ""
if jd_file is not None:
    jd_text = read_any_file(jd_file)

# Warnings if libraries missing
if any((f.name.lower().endswith(".docx") for f in (resumes or []))) and docx is None:
    st.warning("DOCX uploaded but python-docx is not installed. Add `python-docx` to requirements.txt for DOCX support.")
if (jd_file is not None and jd_file.name.lower().endswith(".docx")) and docx is None:
    st.warning("JD is DOCX but python-docx is not installed. Add `python-docx` to requirements.txt.")

if any((f.name.lower().endswith(".pdf") for f in (resumes or []))) and PdfReader is None:
    st.warning("PDF uploaded but pypdf is not installed. Add `pypdf` to requirements.txt for PDF support.")
if (jd_file is not None and jd_file.name.lower().endswith(".pdf")) and PdfReader is None:
    st.warning("JD is PDF but pypdf is not installed. Add `pypdf` to requirements.txt.")

# Run button
run = st.button("🤖 Run TalentPulse Matching")

if run:
    if not resumes or len(resumes) == 0:
        st.error("Please upload at least 1 resume.")
        st.stop()
    if not jd_text.strip():
        st.error("Please upload a Job Description file (or ensure it contains readable text).")
        st.stop()

    results = []
    jd_sk = extract_skills(jd_text)

    for rf in resumes:
        text = read_any_file(rf)
        if not text.strip():
            candidate_name = rf.name  # fallback to filename
            score = 0
            matched = []
            missing = jd_sk
        else:
            candidate_name = guess_candidate_name(text)
            score = ai_similarity_score(text, jd_text)
            res_sk = extract_skills(text)
            matched = sorted(list(set(res_sk) & set(jd_sk)))
            missing = sorted(list(set(jd_sk) - set(res_sk)))

        results.append({
            "Candidate": candidate_name,
            "Resume File": rf.name,
            "Match Score (%)": score,
            "Match Level": label_from_score(score),
            "Matched Skills": ", ".join(matched) if matched else "-",
            "Missing Skills": ", ".join(missing) if missing else "-"
        })

    df = pd.DataFrame(results).sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

    st.subheader("📊 Ranking Results")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 Explainable AI (XAI)")
    st.info(
    )
