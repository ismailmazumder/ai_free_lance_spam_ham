# use_model.py
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string

# Ensure NLTK deps
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# লোড ভেক্টরাইজার ও মডেল
if not os.path.exists('tfidf_vectorizer.pkl') or not os.path.exists('naive_bayes_model.pkl'):
    raise FileNotFoundError("Run train.py first to generate the .pkl files")

tf = joblib.load('tfidf_vectorizer.pkl')
clf = joblib.load('naive_bayes_model.pkl')

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(lemmatizer.lemmatize(w) for w in words)
