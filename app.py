import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

st.set_page_config(page_title="TalentPulse ATS", page_icon="🤖", layout="wide")

# -------------------- STYLE --------------------
st.markdown("""
<style>
.big {font-size:40px;font-weight:800;margin-bottom:0px}
.muted {color:#94a3b8;margin-top:0px}
.card {background:#0b1220;border:1px solid #1f2a44;padding:16px;border-radius:14px}
.badge {display:inline-block;padding:6px 12px;border-radius:999px;font-size:13px;margin:4px 6px 0 0;color:white;background:#2563eb}
.ok {background:#22c55e}
.warn {background:#f59e0b}
.bad {background:#ef4444}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big">🤖 TalentPulse ATS (POC)</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Interactive screening prototype: upload resumes + JD → score → rank → filter → shortlist</div>', unsafe_allow_html=True)
st.markdown("---")

# -------------------- DATABASES --------------------
SKILLS_DB = [
    "python","java","sql","machine learning","data analysis","analytics",
    "cloud","aws","azure","communication","reporting","dashboard",
    "stakeholder","excel","statistics","power bi","tableau","docker","kubernetes",
    "javascript","react","node","git","linux","api","microservices"
]

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\-\.@]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_skills(text: str) -> list:
    t = clean_text(text)
    found = []
    for s in SKILLS_DB:
        if s in t:
            found.append(s)
    return sorted(list(set(found)))

def ai_similarity_score(resume: str, jd: str) -> int:
    resume_c = clean_text(resume)
    jd_c = clean_text(jd)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume_c, jd_c])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return int(round(sim * 100))

def label_from_score(score: int) -> str:
    if score >= 80: return "Excellent"
    if score >= 60: return "Strong"
    if score >= 40: return "Moderate"
    return "Weak"

def recommendation(score: int, knocked_out: bool) -> str:
    if knocked_out:
        return "Reject (Knockout)"
    if score >= 80:
        return "Shortlist"
    if score >= 60:
        return "Next Round"
    if score >= 40:
        return "Hold"
    return "Reject"

# -------------------- FILE READERS --------------------
def read_txt(f) -> str:
    return f.read().decode("utf-8", errors="ignore")

def read_docx(f) -> str:
    if docx is None:
        return ""
    d = docx.Document(f)
    return "\n".join([p.text for p in d.paragraphs])

def read_pdf(f) -> str:
    if PdfReader is None:
        return ""
    reader = PdfReader(f)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

def read_any_file(f) -> str:
    n = f.name.lower()
    if n.endswith(".txt"): return read_txt(f)
    if n.endswith(".docx"): return read_docx(f)
    if n.endswith(".pdf"): return read_pdf(f)
    return ""

# -------------------- PARSING --------------------
def filename_to_name(filename: str) -> str:
    base = re.sub(r"\.(pdf|docx|txt)$", "", filename, flags=re.I)
    base = base.replace("_", " ").replace("-", " ")
    base = re.sub(r"\s+", " ", base).strip()
    # remove common junk words
    base = re.sub(r"\b(resume|cv)\b", "", base, flags=re.I).strip()
    base = re.sub(r"\s+", " ", base).strip()
    return base if base else "Candidate"

def guess_candidate_name(resume_text: str, filename: str) -> str:
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    top = lines[:10]

    blacklist = {"resume","curriculum vitae","cv","profile","contact","summary","top skills","skills"}
    for ln in top:
        low = ln.lower()
        if any(b in low for b in blacklist):
            continue

        ln2 = re.sub(r"\S+@\S+", "", ln).strip()
        ln2 = re.sub(r"(\+?\d[\d\s\-\(\)]{7,})", "", ln2).strip()
        if re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+){1,3}", ln2):
            return ln2

    return filename_to_name(filename)

def extract_email(text: str) -> str:
    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return m.group(0) if m else "-"

def extract_phone(text: str) -> str:
    m = re.search(r"(\+?\d[\d\s\-\(\)]{7,}\d)", text)
    return m.group(0) if m else "-"

def extract_years(text: str) -> str:
    # finds patterns like "5 years", "3+ years"
    m = re.search(r"(\d{1,2}\+?)\s+years", text.lower())
    return m.group(1) if m else "-"

# -------------------- UI: UPLOAD --------------------
st.subheader("📤 Upload Resumes + Job Description")

c1, c2 = st.columns(2)
with c1:
    resumes = st.file_uploader("Resumes (TXT / DOCX / PDF) — upload 1 or more", type=["txt","docx","pdf"], accept_multiple_files=True)
with c2:
    jd_file = st.file_uploader("Job Description (TXT / DOCX / PDF) — upload 1", type=["txt","docx","pdf"], accept_multiple_files=False)

if any((f.name.lower().endswith(".docx") for f in (resumes or []))) and docx is None:
    st.warning("DOCX uploaded but `python-docx` not installed. Add it to requirements.txt if you want DOCX support.")
if any((f.name.lower().endswith(".pdf") for f in (resumes or []))) and PdfReader is None:
    st.warning("PDF uploaded but `pypdf` not installed. Add it to requirements.txt if you want PDF support.")

# -------------------- ATS SETTINGS --------------------
st.markdown("---")
st.subheader("⚙️ ATS Screening Settings")

must_have = st.multiselect("Must-have skills (Knockout rules)", options=SKILLS_DB, default=[])
min_score = st.slider("Minimum match score filter", 0, 100, 0)
skill_filter = st.selectbox("Filter by skill (optional)", ["(none)"] + SKILLS_DB)

run = st.button("🚀 Run TalentPulse ATS Screening")

# -------------------- RUN --------------------
if run:
    if not resumes:
        st.error("Upload at least 1 resume.")
        st.stop()
    if jd_file is None:
        st.error("Upload a Job Description file.")
        st.stop()

    jd_text = read_any_file(jd_file)
    if not jd_text.strip():
        st.error("JD file has no readable text. Try TXT/DOCX or a different PDF.")
        st.stop()

    jd_sk = extract_skills(jd_text)

    rows = []
    for rf in resumes:
        r_text = read_any_file(rf)

        # if unreadable resume
        if not r_text.strip():
            name = filename_to_name(rf.name)
            score = 0
            r_sk = []
        else:
            name = guess_candidate_name(r_text, rf.name)
            score = ai_similarity_score(r_text, jd_text)
            r_sk = extract_skills(r_text)

        matched = sorted(list(set(r_sk) & set(jd_sk)))
        missing = sorted(list(set(jd_sk) - set(r_sk)))

        # knockout
        knocked_out = any(skill not in r_sk for skill in must_have) if must_have else False

        rows.append({
            "Candidate": name,
            "Resume File": rf.name,
            "Email": extract_email(r_text),
            "Phone": extract_phone(r_text),
            "Years (approx.)": extract_years(r_text),
            "Match Score (%)": score,
            "Match Level": label_from_score(score),
            "Knockout": "Yes" if knocked_out else "No",
            "Recommendation": recommendation(score, knocked_out),
            "Matched Skills": ", ".join(matched) if matched else "-",
            "Missing Skills": ", ".join(missing) if missing else "-"
        })

    df = pd.DataFrame(rows).sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

    # apply filters
    df_view = df[df["Match Score (%)"] >= min_score].copy()
    if skill_filter != "(none)":
        df_view = df_view[df_view["Matched Skills"].str.contains(skill_filter, case=False, na=False)]

    st.markdown("---")
    st.subheader("📊 ATS Ranking Results")

    st.dataframe(df_view, use_container_width=True)

    # export
    st.download_button(
        "⬇️ Download Results CSV",
        data=df_view.to_csv(index=False).encode("utf-8"),
        file_name="talentpulse_ats_results.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("🧾 Candidate Detail View (click-friendly)")
    candidate_names = df_view["Candidate"].tolist()
    if candidate_names:
        selected = st.selectbox("Select a candidate to view details", candidate_names)
        row = df_view[df_view["Candidate"] == selected].iloc[0]

        colA, colB = st.columns([1.2, 1])

        with colA:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### {row['Candidate']}")
            st.write(f"**Recommendation:** {row['Recommendation']}")
            st.write(f"**Match Score:** {row['Match Score (%)']}% ({row['Match Level']})")
            st.write(f"**Knockout:** {row['Knockout']}")
            st.write(f"**Email:** {row['Email']}")
            st.write(f"**Phone:** {row['Phone']}")
            st.write(f"**Years (approx.):** {row['Years (approx.)']}")
            st.markdown("</div>", unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Skill Summary")
            st.write("✅ Matched skills:")
            st.write(row["Matched Skills"])
            st.write("❌ Missing skills:")
            st.write(row["Missing Skills"])
            st.markdown("</div>", unsafe_allow_html=True)

        # notes (session)
        st.markdown("---")
        st.subheader("📝 Recruiter Notes (session-based)")
        key = f"note_{selected}"
        st.session_state[key] = st.text_area("Add notes for this candidate", value=st.session_state.get(key, ""), height=120)
        st.caption("Notes are saved during this session for demo purposes (POC).")
    else:
        st.info("No candidates match the current filters. Lower the minimum score or remove the skill filter.")
