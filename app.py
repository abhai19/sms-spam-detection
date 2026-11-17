"""
app.py
One-page Streamlit app: paste an SMS, click "Check Message", see Spam / Legitimate.
The app only LOADS the saved model; it never trains.
"""

import json
from pathlib import Path

import joblib
import streamlit as st

# Paths relative to this file, so they work on Mac and on Streamlit Cloud
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "spam_pipeline.joblib"
METRICS_PATH = BASE_DIR / "results" / "metrics.json"

SPAM_SAMPLE = "Congratulations! You have won a free prize. Reply WIN now to claim your reward."
LEGIT_SAMPLE = "Hi, are we still meeting at the library at 5 pm today?"


@st.cache_resource  # load the model once, not on every click
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def set_text(text):
    """Called when a sample button is clicked: fills the text box."""
    st.session_state["sms_text"] = text


st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
st.title("📩 SMS Spam Detector")
st.caption("Cloud-Based SMS Spam Detection Using Machine Learning (TF-IDF + Naive Bayes)")

model = load_model()
if model is None:
    st.error("Trained model not found. Run `python train_model.py` first, "
             "then make sure `model/spam_pipeline.joblib` is in the project.")
    st.stop()

st.text_area("Paste an SMS message:", key="sms_text", height=120)

col1, col2 = st.columns(2)
col1.button("Try a spam sample", on_click=set_text, args=(SPAM_SAMPLE,))
col2.button("Try a legitimate sample", on_click=set_text, args=(LEGIT_SAMPLE,))

if st.button("Check Message", type="primary"):
    message = st.session_state.get("sms_text", "").strip()
    if not message:
        st.warning("Please type or paste a message first.")
    else:
        prediction = model.predict([message])[0]      # 1 = spam, 0 = legitimate
        spam_prob = model.predict_proba([message])[0][1]
        if prediction == 1:
            st.error("🚫 Spam")
        else:
            st.success("✅ Legitimate")
        st.caption(f"Model's spam probability: {spam_prob:.1%}")

# Show the measured test results saved by train_model.py (if present)
if METRICS_PATH.exists():
    with st.expander("Model evaluation (20% held-out test set)"):
        m = json.loads(METRICS_PATH.read_text())
        st.write(f"Accuracy: **{m['accuracy']:.2%}** · "
                 f"Spam precision: **{m['spam_precision']:.2%}** · "
                 f"Spam recall: **{m['spam_recall']:.2%}** · "
                 f"Spam F1: **{m['spam_f1']:.2%}**")

st.divider()
st.caption("Dataset: UCI SMS Spam Collection (Almeida & Hidalgo, 2011). "
           "A student project; predictions are not perfect.")
