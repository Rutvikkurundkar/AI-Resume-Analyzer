import streamlit as st
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline
import torch
import torch.nn.functional as F
import numpy as np
import joblib
import fitz  # PyMuPDF

# ✅ Must be first Streamlit command
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# === Mean Pooling ===
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# === Load HuggingFace Embedding Model ===
@st.cache_resource
def load_hf_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

tokenizer, hf_model = load_hf_model()

# === Load HuggingFace LLM for tailoring resume ===
@st.cache_resource
def load_text_generator():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

text_generator = load_text_generator()

# === Load Custom Trained Regression Model ===
@st.cache_resource
def load_custom_model():
    return joblib.load("resume_regressor.pkl")

regression_model = load_custom_model()

# === Embedding Function ===
def get_embedding(text):
    encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = hf_model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(embedding, p=2, dim=1).numpy()[0]

# === Extract Resume Text ===
def extract_resume_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        try:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {e}")
            return ""
    else:
        st.error("❌ Unsupported file type.")
        return ""

# === Tailor Resume with Hugging Face LLM ===
def tailor_resume_hf(resume_text, job_description):
    prompt = f"""
You are a professional resume editor. Improve the resume below to better match the given job description.
Focus on relevant skills, clarity, structure, and language alignment.

Job Description:
{job_description}

Resume:
{resume_text}

Provide only the improved resume.
"""
    try:
        response = text_generator(prompt, max_length=512, do_sample=False)[0]['generated_text']
        return response.strip()
    except Exception as e:
        return f"❌ Error tailoring resume: {e}"

# === Streamlit UI ===
st.markdown("""
    <style>
    .reportview-container .markdown-text-container {
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-size: 1em;
    }
    .stTextArea, .stFileUploader {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI Resume Analyzer + Tailoring Assistant")
st.markdown("Analyze your resume against a job description and get a tailored version using Hugging Face LLM.")

jd_text = st.text_area("📄 Paste the Job Description", height=200)
uploaded_resume = st.file_uploader("📤 Upload Resume (PDF or TXT)", type=["pdf", "txt"])

# Model Selection
mode = st.radio("🔍 Choose Matching Method", ["Custom ML Model (pkl)", "Cosine Similarity"])

col1, col2 = st.columns(2)

with col1:
    if st.button("🔎 Analyze Match"):
        if not jd_text or not uploaded_resume:
            st.warning("⚠️ Please provide both a Job Description and Resume.")
        else:
            resume_text = extract_resume_text(uploaded_resume)
            if resume_text:
                jd_embed = get_embedding(jd_text)
                resume_embed = get_embedding(resume_text)

                if mode == "Custom ML Model (pkl)":
                    input_vector = np.abs(jd_embed - resume_embed).reshape(1, -1)
                    try:
                        score = regression_model.predict(input_vector)[0]
                        st.success(f"✅ Custom Model Match Score: {round(score * 10, 2)}%")
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
                else:
                    similarity = torch.nn.functional.cosine_similarity(
                        torch.tensor(jd_embed), torch.tensor(resume_embed), dim=0
                    ).item()
                    st.success(f"🔁 Cosine Similarity Match Score: {round(similarity * 100, 2)}%")
            else:
                st.error("❌ Could not extract text from the resume.")

with col2:
    if st.button("✏️ Tailor My Resume"):
        if not jd_text or not uploaded_resume:
            st.warning("⚠️ Please provide both a Job Description and Resume.")
        else:
            resume_text = extract_resume_text(uploaded_resume)
            if resume_text:
                tailored = tailor_resume_hf(resume_text, jd_text)
                st.text_area("🎯 Tailored Resume", tailored, height=400)
                st.download_button("⬇️ Download Tailored Resume", tailored, file_name="tailored_resume.txt")
            else:
                st.error("❌ Could not extract text from the resume.")
