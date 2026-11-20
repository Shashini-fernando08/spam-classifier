import re
import joblib
import streamlit as st


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.caption("TF-IDF + Machine Learning | Trained on the SMS Spam Collection dataset")

model = load_model()
msg = st.text_area("Enter a message:", height=150,
                   placeholder="Congratulations! You won a free prize. Click here to claim...")

if st.button("Check"):
    if not msg.strip():
        st.warning("Please enter a message.")
    else:
        proba = model.predict_proba([clean(msg)])[0][1]
        if proba >= 0.5:
            st.error(f"🚨 SPAM  (probability: {proba:.1%})")
        else:
            st.success(f"✅ NOT SPAM  (spam probability: {proba:.1%})")
        st.progress(float(proba))
