import streamlit as st
import pandas as pd

st.set_page_config(page_title="TalentPulse", layout="wide")

st.title("🤖 TalentPulse – AI Resume Matching Prototype")
st.subheader("Smart Talent Intelligence Dashboard")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("📄 Resume Input")
    resume_text = st.text_area("Paste Candidate Resume Text Here", height=200)

with col2:
    st.header("🧾 Job Description Input")
    jd_text = st.text_area("Paste Job Description Here", height=200)

st.markdown("---")

def extract_skills(text):
    skills_db = [
        "python", "java", "sql", "machine learning",
        "data analysis", "cloud", "aws", "communication"
    ]
    found = []
    for skill in skills_db:
        if skill.lower() in text.lower():
            found.append(skill)
    return found

if st.button("🔍 Run TalentPulse Matching Engine"):

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))

    score = int((len(matched) / len(jd_skills)) * 100) if len(jd_skills) else 0

    st.success("Matching Completed!")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("🎯 Match Score", f"{score}%")

    with col4:
        st.write("✅ Matched Skills")
        st.write(matched if matched else "No matched skills found")

    with col5:
        st.write("❌ Missing Skills")
        st.write(missing if missing else "No missing skills")

    st.markdown("---")
    st.header("🧠 AI Explanation Panel (XAI)")

    explanation = f"""
    The system analysed the resume and job description using keyword skill matching.

    Total job skills detected: {len(jd_skills)}
    Skills matched: {len(matched)}

    Final Match Score = (Matched Skills / Required Skills) × 100
    Final Score = {score}%
    """
    st.info(explanation)

    st.header("📊 Candidate Ranking Simulation")

    df = pd.DataFrame({
        "Candidate": ["Candidate A", "Candidate B", "Candidate C"],
        "Match Score (%)": [score, 78, 62],
        "Status": ["You", "Strong Match", "Moderate Match"]
    })

    st.table(df)

