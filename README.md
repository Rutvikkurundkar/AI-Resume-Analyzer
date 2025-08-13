# 🧠 AI Resume Analyzer & Tailoring Assistant

An AI-powered application built with **Streamlit** that:
- Analyzes your resume against a given job description
- Calculates a match score using **ML models** or **cosine similarity**
- Suggests improvements to better align with the job
- Generates a tailored resume version using **Hugging Face LLM**

---

## 🚀 Features
- **Resume Parsing** — Extracts text from PDF and TXT resumes using `PyMuPDF`.
- **Skill & Content Matching** — Uses embeddings from `sentence-transformers/all-MiniLM-L6-v2`.
- **Match Score Calculation** — Two modes:
  - **Custom ML Regression Model** (`resume_regressor.pkl`)
  - **Cosine Similarity** between embeddings
- **Resume Tailoring** — Improves resume content using `google/flan-t5-large`.
- **User-Friendly Interface** — Built with Streamlit for easy interaction.

---

## 🛠️ Tech Stack
- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Backend / AI**:
  - Hugging Face Transformers
  - Sentence Transformers (MiniLM)
  - scikit-learn
- **File Parsing**: PyMuPDF
- **Model Storage**: Git LFS for large `.pkl` files
- **Language**: Python 3.x

---

## 📂 Project Structure
AI-Resume-Analyzer/
│-- app.py # Main Streamlit app
│-- requirements.txt # Python dependencies
│-- embedding_model.pkl # Embedding model (large file via Git LFS)
│-- resume_regressor.pkl # Custom ML regression model (via Git LFS)
│-- README.md # Documentation
│-- data/ # (Optional) Sample resumes/job descriptions

yaml

---

## ⚡ Installation & Usage

### 1️⃣ Clone the repository
Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Streamlit app
bash
Copy
Edit
streamlit run app.py



<img width="1912" height="892" alt="image" src="https://github.com/user-attachments/assets/5c517ba8-e855-429b-b1e1-1633df02d7c3" />

