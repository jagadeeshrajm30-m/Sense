import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st

from src.predict import predict_sentiment

st.set_page_config(
    page_title="ReviewSense - Sentiment Analysis",
    page_icon="⭐",
    layout="centered",
)

st.title("🧠 ReviewSense – Sentiment Analysis")
st.write(
    "Type a review below and I'll predict whether it's **positive** or **negative**."
)

with st.form("sentiment_form"):
    text = st.text_area(
        "Enter your review text",
        height=200,
        placeholder="Example: The movie was absolutely fantastic and I loved every moment!",
    )
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(text)

        label = result["label_name"]
        confidence = result["confidence"]

        if label == "positive":
            st.success(f"✅ Sentiment: **Positive** (confidence: {confidence:.2%})")
        else:
            st.error(f"❌ Sentiment: **Negative** (confidence: {confidence:.2%})")

        with st.expander("View raw output"):
            st.json(result)

st.markdown("---")
st.caption(
    "Built with scikit-learn + TF-IDF. You can extend this later with transformer models (BERT, etc.)."
)
