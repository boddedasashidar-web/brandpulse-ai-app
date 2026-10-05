import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import plotly.express as px
import datetime

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.set_page_config(page_title="BrandPulse AI", page_icon="📊", layout="wide")

MAX_LEN = 40

# ---------- Load models (cached so they only load once) ----------
@st.cache_resource
def load_all_models():
    lr_model = joblib.load('logistic_regression_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    lstm_model = load_model('best_lstm_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return lr_model, tfidf, lstm_model, tokenizer

lr_model, tfidf, lstm_model, tokenizer = load_all_models()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
label_names = ['negative', 'neutral', 'positive']

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

def predict_lstm(text):
    cleaned = clean_tweet(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = lstm_model.predict(padded, verbose=0)[0]
    return label_names[np.argmax(probs)], float(np.max(probs)), probs

def predict_lr(text):
    cleaned = clean_tweet(text)
    vec = tfidf.transform([cleaned])
    pred = lr_model.predict(vec)[0]
    probs = lr_model.predict_proba(vec)[0]
    return pred, float(np.max(probs))

# ---------- Session state: store prediction history for the dashboard ----------
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['time', 'text', 'sentiment'])

# ---------- UI ----------
st.title("📊 BrandPulse AI")
st.caption("Real-time sentiment monitoring dashboard — Twitter/X brand mentions")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analyze a tweet")
    user_text = st.text_area("Paste a tweet or customer message:", height=100,
                               placeholder="e.g. The service was amazing!")
    model_choice = st.radio("Model:", ["LSTM (Deep Learning)", "Logistic Regression (Classical)"], horizontal=True)

    if st.button("Analyze Sentiment", type="primary") and user_text.strip():
        if model_choice.startswith("LSTM"):
            sentiment, confidence, probs = predict_lstm(user_text)
        else:
            sentiment, confidence = predict_lr(user_text)

        color = {"positive": "green", "negative": "red", "neutral": "gray"}[sentiment]
        st.markdown(f"### Sentiment: :{color}[{sentiment.upper()}]")
        st.progress(confidence)
        st.caption(f"Confidence: {confidence*100:.1f}%")

        # log to session history for the dashboard
        new_row = pd.DataFrame([{
            'time': datetime.datetime.now(),
            'text': user_text[:60],
            'sentiment': sentiment
        }])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

with col2:
    st.subheader("Try examples")
    examples = [
        "The service was amazing!",
        "I waited 4 hours just to get a cold burger.",
        "My flight is scheduled for tomorrow at 9am."
    ]
    for ex in examples:
        st.code(ex, language=None)

st.divider()

# ---------- Dashboard: distribution + trend ----------
st.subheader("Session sentiment dashboard")

if len(st.session_state.history) > 0:
    d1, d2 = st.columns(2)

    with d1:
        dist = st.session_state.history['sentiment'].value_counts().reset_index()
        dist.columns = ['sentiment', 'count']
        fig_pie = px.pie(dist, names='sentiment', values='count', title="Sentiment Distribution",
                          color='sentiment',
                          color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with d2:
        trend = st.session_state.history.copy()
        trend['sentiment_score'] = trend['sentiment'].map({'negative': -1, 'neutral': 0, 'positive': 1})
        fig_line = px.line(trend, x='time', y='sentiment_score', markers=True,
                            title="Sentiment Trend (this session)")
        fig_line.update_yaxes(tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive'])
        st.plotly_chart(fig_line, use_container_width=True)

    st.dataframe(st.session_state.history.tail(10), use_container_width=True)
else:
    st.info("Analyze a few tweets above to populate the dashboard.")