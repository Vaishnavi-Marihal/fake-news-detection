import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def predict_fake_news(text):
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0].max()
    result = "REAL" if pred == 1 else "FAKE"
    return result, proba

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Enter any news headline or article below to check if it's real or fake.")

user_input = st.text_area("📝 Paste news text here:", height=150)

if st.button("🔍 Check for Fake News"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        result, confidence = predict_fake_news(user_input)
        if result == "REAL":
            st.success(f"✅ **Prediction: REAL NEWS**")
        else:
            st.error(f"❌ **Prediction: FAKE NEWS**")
        st.write(f"**Confidence:** {confidence:.4f}")

st.sidebar.header("About")
st.sidebar.write("""
This app uses machine learning to detect fake news.
Built with Python, Scikit-learn, and Streamlit.
""")
st.sidebar.write("© 2025 Fake News Detector")
