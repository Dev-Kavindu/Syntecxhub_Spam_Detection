import os
import pickle
import re

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


st.set_page_config(
    page_title="Email Shield | Spam Detector",
    page_icon="Shield",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def download_nltk_data() -> None:
    resources = {
        "corpora/stopwords": "stopwords",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for path, package_name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package_name, quiet=True)


download_nltk_data()


@st.cache_resource(show_spinner=False)
def load_resources():
    missing_files = [name for name in ("vectorizer.pkl", "spam_model.pkl") if not os.path.exists(name)]
    if missing_files:
        st.error(
            "Missing model file(s): **" + ", ".join(missing_files) + "**\n\n"
            "Place `vectorizer.pkl` and `spam_model.pkl` in the same folder as `app.py`."
        )
        st.stop()

    try:
        with open("vectorizer.pkl", "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        with open("spam_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        return vectorizer, model
    except Exception as exc:
        st.error(f"Failed to load model assets: {exc}")
        st.stop()


vectorizer, model = load_resources()

_stemmer = PorterStemmer()
_stopwords = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = [
        _stemmer.stem(token)
        for token in text.split()
        if token not in _stopwords
    ]
    return " ".join(tokens)


def analyze_message(message: str) -> dict:
    cleaned_message = clean_text(message)
    vectorized_message = vectorizer.transform([cleaned_message])
    prediction = int(model.predict(vectorized_message)[0])

    try:
        probabilities = model.predict_proba(vectorized_message)[0]
        ham_confidence = float(probabilities[0])
        spam_confidence = float(probabilities[1])
    except AttributeError:
        spam_confidence = 0.93 if prediction == 1 else 0.07
        ham_confidence = 1.0 - spam_confidence

    return {
        "prediction": prediction,
        "cleaned_message": cleaned_message,
        "ham_confidence": ham_confidence,
        "spam_confidence": spam_confidence,
    }


def set_message(text: str) -> None:
    st.session_state.msg_input = text
    st.session_state.last_result = None


def clear_message() -> None:
    st.session_state.msg_input = ""
    st.session_state.last_result = None


def run_analysis() -> None:
    message = st.session_state.msg_input.strip()
    if not message:
        st.session_state.last_result = {"error": "Please paste some text before clicking Analyze."}
        return

    st.session_state.last_result = analyze_message(message)


if "msg_input" not in st.session_state:
    st.session_state.msg_input = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --primary: #0ea5e9;
    --primary-dark: #0284c7;
    --primary-light: #06b6d4;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-tertiary: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --border: #475569;
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.5);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.35);
    --shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.4);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

* {
    box-sizing: border-box;
}

html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary);
    background: var(--bg-primary);
}

body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

header, footer, #MainMenu {
    visibility: hidden;
}

.block-container {
    max-width: 1400px !important;
    padding-top: 3rem !important;
    padding-bottom: 3rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* ========== HEADER SECTION ========== */
.header-section {
    margin-bottom: 3rem;
}

.logo-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(6, 182, 212, 0.1));
    border: 1px solid rgba(14, 165, 233, 0.3);
    border-radius: 999px;
    color: var(--primary-light);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.header-title {
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 900;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    margin: 0 0 0.8rem 0;
    line-height: 1.1;
}

.header-subtitle {
    font-size: 1.125rem;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.6;
    max-width: 700px;
    font-weight: 400;
}

/* ========== CARDS & PANELS ========== */
.card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}

.card:hover {
    border-color: var(--primary);
    box-shadow: var(--shadow-md);
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.05), rgba(6, 182, 212, 0.03));
}

.card-section {
    margin-bottom: 2rem;
}

.section-title {
    font-size: 0.875rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin: 0 0 1rem 0;
}

/* ========== METRICS GRID ========== */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2.5rem;
}

.metric {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    transition: all 0.3s ease;
}

.metric:hover {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.08), rgba(6, 182, 212, 0.05));
    border-color: var(--primary);
    box-shadow: var(--shadow-md);
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--primary-light);
    margin-bottom: 0.5rem;
}

.metric-note {
    font-size: 0.875rem;
    color: var(--text-muted);
}

/* ========== MAIN LAYOUT ========== */
.main-container {
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
}

.input-section {
    grid-column: 1;
}

.summary-section {
    grid-column: 2;
}

/* ========== INPUT AREA ========== */
.input-label {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
}

.input-description {
    font-size: 0.875rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

div[data-testid="stTextArea"] textarea {
    min-height: 280px !important;
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9375rem !important;
    line-height: 1.7 !important;
    padding: 1.25rem !important;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: var(--text-muted) !important;
    opacity: 0.8 !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--primary) !important;
    outline: none !important;
    box-shadow:
        inset 0 1px 2px rgba(0, 0, 0, 0.3),
        0 0 0 3px rgba(14, 165, 233, 0.15) !important;
    background: rgba(51, 65, 85, 0.8) !important;
}

/* ========== ACTION BUTTONS ========== */
.button-group {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin: 1.5rem 0 2rem 0;
}

.stButton > button {
    min-height: 48px !important;
    border-radius: var(--radius-lg) !important;
    font-weight: 600 !important;
    font-size: 0.9375rem !important;
    border: none !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    box-shadow: 0 8px 24px rgba(14, 165, 233, 0.3) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(14, 165, 233, 0.4) !important;
}

.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

.stButton > button[kind="secondary"] {
    background: rgba(14, 165, 233, 0.1) !important;
    color: var(--primary-light) !important;
    border: 1px solid var(--primary) !important;
}

.stButton > button[kind="secondary"]:hover {
    background: rgba(14, 165, 233, 0.15) !important;
}

/* ========== SAMPLE CARDS ========== */
.samples-section {
    margin-top: 2rem;
}

.sample-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
}

.sample-description {
    font-size: 0.8125rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
    line-height: 1.5;
}

.sample-buttons {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
}

/* ========== SUMMARY PANEL ========== */
.summary-card {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.stat-box {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    transition: all 0.3s ease;
}

.stat-box:hover {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.08), rgba(6, 182, 212, 0.05));
    border-color: var(--primary);
}

.stat-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}

.stat-value {
    font-size: 1.875rem;
    font-weight: 800;
    color: var(--primary-light);
}

.info-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-top: 1rem;
}

.info-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
}

.info-description {
    font-size: 0.875rem;
    color: var(--text-muted);
    line-height: 1.6;
    margin: 0;
}

/* ========== RESULT CARD ========== */
.result-container {
    margin-top: 2rem;
}

.result-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2rem;
    box-shadow: var(--shadow-md);
}

.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1.5rem;
}

.badge-spam {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.badge-ham {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.result-title {
    font-size: 1.875rem;
    font-weight: 800;
    margin: 0 0 1rem 0;
    letter-spacing: -0.02em;
}

.title-spam {
    color: #fca5a5;
}

.title-ham {
    color: #6ee7b7;
}

.result-description {
    font-size: 1rem;
    color: var(--text-secondary);
    line-height: 1.7;
    margin-bottom: 2rem;
}

.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2.5rem;
}

.result-stat {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.08), rgba(6, 182, 212, 0.05));
    border: 1px solid rgba(14, 165, 233, 0.2);
    border-radius: var(--radius-lg);
    padding: 1.75rem;
    text-align: center;
    transition: all 0.3s ease;
}

.result-stat:hover {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(6, 182, 212, 0.1));
    border-color: var(--primary);
    box-shadow: var(--shadow-md);
    transform: translateY(-4px);
}

.result-stat-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 1rem;
}

.result-stat-value {
    font-size: 2.25rem;
    font-weight: 800;
    color: var(--primary-light);
    line-height: 1;
}

.progress-section {
    margin-top: 2rem;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.75rem;
}

.progress-label {
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.confidence-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
}

.confidence-item {
    text-align: center;
    padding: 1.5rem;
    background: rgba(14, 165, 233, 0.05);
    border-radius: var(--radius-md);
    border: 1px solid rgba(14, 165, 233, 0.1);
}

.confidence-label {
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}

.confidence-value {
    font-size: 2.5rem;
    font-weight: 900;
    color: var(--primary-light);
    line-height: 1;
}

/* ========== METRICS ========== */
[data-testid="metric-container"] {
    background-color: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(14, 165, 233, 0.2) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

[data-testid="metric-container"] > div:nth-child(1) {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

[data-testid="metric-container"] > div:nth-child(2) {
    color: #06b6d4 !important;
    font-size: 2.25rem !important;
    font-weight: 900 !important;
}

/* Custom card styling */
.metric-card {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.08), rgba(6, 182, 212, 0.05));
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-radius: 14px;
    padding: 1.75rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
}

.metric-card:hover {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(6, 182, 212, 0.1));
    border-color: rgba(14, 165, 233, 0.4);
    box-shadow: 0 8px 24px rgba(14, 165, 233, 0.2);
}

.metric-card-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #cbd5e1;
    margin-bottom: 0.75rem;
    display: block;
}

.metric-card-value {
    font-size: 2.5rem;
    font-weight: 900;
    color: #06b6d4;
    line-height: 1;
}

/* ========== EXPANDER ========== */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--bg-secondary) !important;
    margin-top: 1.5rem !important;
}

[data-testid="stExpander"] button {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.9375rem !important;
}

[data-testid="stExpander"] button:hover {
    color: var(--primary) !important;
}

/* ========== CODE BLOCKS ========== */
[data-testid="stCodeBlock"] {
    margin-top: 1rem !important;
}

[data-testid="stCodeBlock"] > div {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9)) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}

/* ========== ALERTS ========== */
[data-testid="stAlert"] {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border) !important;
    margin-top: 1rem !important;
}

/* ========== FOOTER ========== */
.footer {
    margin-top: 4rem;
    padding: 2.5rem 2rem;
    background: linear-gradient(180deg, rgba(14, 165, 233, 0.05) 0%, rgba(6, 182, 212, 0.03) 100%);
    border-top: 2px solid rgba(14, 165, 233, 0.2);
    border-radius: var(--radius-lg);
    text-align: center;
    color: #f1f5f9;
    font-size: 0.9375rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    transition: all 0.3s ease;
}

.footer:hover {
    background: linear-gradient(180deg, rgba(14, 165, 233, 0.1) 0%, rgba(6, 182, 212, 0.08) 100%);
    border-color: rgba(14, 165, 233, 0.35);
    box-shadow: 0 8px 24px rgba(14, 165, 233, 0.1);
}

.footer-text {
    color: #f1f5f9;
    margin: 0;
    line-height: 1.6;
}

.footer-author {
    color: #06b6d4;
    font-weight: 600;
}

/* ========== RESPONSIVE ========== */
@media (max-width: 1024px) {
    .main-container {
        grid-template-columns: 1fr;
    }

    .input-section {
        grid-column: 1;
    }

    .summary-section {
        grid-column: 1;
    }
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .header-title {
        font-size: 1.875rem;
    }

    .metrics-grid {
        grid-template-columns: 1fr;
    }

    .button-group {
        grid-template-columns: 1fr;
    }

    .result-grid {
        grid-template-columns: 1fr;
    }

    .card {
        padding: 1.25rem;
    }

    .metric-card-value {
        font-size: 2rem !important;
    }
}

@media (max-width: 480px) {
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    .header-title {
        font-size: 1.5rem;
    }

    .header-subtitle {
        font-size: 0.9375rem;
    }

    div[data-testid="stTextArea"] textarea {
        min-height: 200px !important;
        padding: 1rem !important;
    }

    .result-card {
        padding: 1.25rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)




st.markdown(
    """
<div class="header-section">
    <div class="logo-badge">✓ Email Shield</div>
    <h1 class="header-title">Professional Spam Detection</h1>
    <p class="header-subtitle">Analyze emails instantly with our advanced filtering system. Get clear verdicts with confidence scores to keep your inbox secure.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="metrics-grid">
        <div class="metric">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">97.29%</div>
            <div class="metric-note">Naive Bayes Classifier</div>
        </div>
        <div class="metric">
            <div class="metric-label">Training Samples</div>
            <div class="metric-value">5,171</div>
            <div class="metric-note">Spam & Ham Messages</div>
        </div>
        <div class="metric">
            <div class="metric-label">Detection Speed</div>
            <div class="metric-value">&lt;50ms</div>
            <div class="metric-note">Per Message</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Create layout structure
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<div class="input-section">', unsafe_allow_html=True)

# Input Section Header
st.markdown(
    """
    <div class="section-title">📝 Message Analysis</div>
    <div class="input-description">
        Paste your email or SMS content below. The model will analyze it using the same processing
        pipeline used during training for accurate spam detection.
    </div>
    """,
    unsafe_allow_html=True,
)

message = st.text_area(
    label="Message content",
    label_visibility="collapsed",
    placeholder="Paste the email or SMS text here...",
    height=280,
    key="msg_input",
)

# Action Buttons
st.markdown('<div class="button-group">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.button("🔍 Analyze message", type="primary", use_container_width=True, on_click=run_analysis)
with col2:
    st.button("🗑️ Clear input", type="secondary", use_container_width=True, on_click=clear_message)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # End input-section

# Summary Section
st.markdown('<div class="summary-section">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-title">📊 Message Summary</div>
    """,
    unsafe_allow_html=True,
)

current_text = st.session_state.msg_input.strip()
current_word_count = len(current_text.split()) if current_text else 0
current_char_count = len(current_text)
current_cleaned = clean_text(current_text) if current_text else ""
current_cleaned_words = len(current_cleaned.split()) if current_cleaned else 0

st.markdown(
    f"""
    <div class="summary-card">
        <div class="stat-box">
            <div class="stat-label">Characters</div>
            <div class="stat-value">{current_char_count}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Words</div>
            <div class="stat-value">{current_word_count}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Cleaned Tokens</div>
            <div class="stat-value">{current_cleaned_words}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-section">
        <div class="info-title">📘 How It Works</div>
        <p class="info-description">
            <strong>Higher spam confidence</strong> means the model detected suspicious patterns.
            <strong>Higher legitimate confidence</strong> means the message appears normal and safe.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)  # End summary-section
st.markdown('</div>', unsafe_allow_html=True)  # End main-container

# Results Section
st.markdown('<div class="result-container">', unsafe_allow_html=True)

result = st.session_state.last_result

if result:
    if result.get("error"):
        st.error(result["error"])
    else:
        prediction = result["prediction"]
        spam_confidence = result["spam_confidence"]
        ham_confidence = result["ham_confidence"]
        cleaned_message = result["cleaned_message"]

        if prediction == 1:
            badge_class = "badge-spam"
            title_class = "title-spam"
            badge_text = "⚠️ Spam Detected"
            title_text = "This message looks risky"
            lead_text = (
                "The classifier flagged this content as spam. Be cautious with links, attachments, and personal data requests."
            )
            primary_score = spam_confidence
            secondary_score = ham_confidence
            verdict_text = "Spam"
        else:
            badge_class = "badge-ham"
            title_class = "title-ham"
            badge_text = "✅ Legitimate"
            title_text = "This message looks safe"
            lead_text = (
                "The classifier did not find suspicious patterns. However, always verify sensitive requests manually."
            )
            primary_score = ham_confidence
            secondary_score = spam_confidence
            verdict_text = "Legitimate"

        # Render header and title
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-badge {badge_class}">{badge_text}</div>
                <div class="result-title {title_class}">{title_text}</div>
                <p class="result-description">{lead_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display metrics in a clean grid with custom styling
        col1, col2, col3 = st.columns(3, gap="small")

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span class="metric-card-label">📊 Primary Score</span>
                    <span class="metric-card-value">{int(primary_score * 100)}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span class="metric-card-label">✓ Verdict</span>
                    <span class="metric-card-value">{verdict_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span class="metric-card-label">📈 Confidence</span>
                    <span class="metric-card-value">{int(primary_score * 100)}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Confidence breakdown section with better styling
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <h3 style="color: #f1f5f9; font-size: 1.25rem; font-weight: 600; margin-bottom: 1.5rem;">
                📊 Detailed Confidence Analysis
            </h3>
            """,
            unsafe_allow_html=True,
        )

        conf_col1, conf_col2 = st.columns(2, gap="small")

        with conf_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span class="metric-card-label">🚨 Spam Probability</span>
                    <span class="metric-card-value">{round(spam_confidence * 100, 1)}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with conf_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span class="metric-card-label">✅ Legitimate Probability</span>
                    <span class="metric-card-value">{round(ham_confidence * 100, 1)}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Progress bar section with proper styling
        st.markdown(
            """
            <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(14, 165, 233, 0.2); border-radius: 12px;">
                <p style="color: #cbd5e1; font-weight: 600; margin-bottom: 1rem; font-size: 0.95rem;">Overall Confidence Score</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(primary_score)

else:
    st.markdown(
        """
        <div class="result-card">
            <div class="result-badge badge-ham">🚀 Ready</div>
            <div class="result-title title-ham">No analysis yet</div>
            <p class="result-description">
                Paste a message above and click "Analyze message" to see your spam detection results here.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)  # End result-container

# Footer
st.markdown(
    """
    <div class="footer">
        <p class="footer-text">
            <span class="footer-author">Developed by Kavindu Chamod</span> | Advanced Spam Detection System
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
