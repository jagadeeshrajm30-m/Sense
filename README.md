# ReviewSense – Sentiment Analysis Web App

An end-to-end sentiment analysis project:

- Binary classification of text reviews (positive / negative)
- TF-IDF + Logistic Regression baseline
- Deployed as a Streamlit web application

## Project Structure

```text
.
├─ app/
│  └─ app.py          # Streamlit app
├─ data/
│  └─ raw/
│     └─ reviews.csv  # input dataset
├─ models/
│  └─ baseline_logreg.joblib  # trained model (generated)
├─ notebooks/
│  └─ 01_eda_and_baseline.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ data_utils.py
│  ├─ train.py
│  └─ predict.py
├─ requirements.txt
└─ README.md
