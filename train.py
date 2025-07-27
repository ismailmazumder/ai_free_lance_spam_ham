import pandas as pd
import string
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess your base CSVs merging into static/data.csv if needed
# For brevity, assuming static/data.csv already exists with v1,v2,label
from use_model import DATA_PATH, preprocess_text

df = pd.read_csv(DATA_PATH)
# If initial dataset not loaded, user can manually populate data.csv

df['v2'] = df['v2'].apply(preprocess_text)

# Vectorize & train
tf = TfidfVectorizer(max_features=3000)
X = tf.fit_transform(df['v2']).toarray()
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save
joblib.dump(tf, 'tfidf_vectorizer.pkl')
joblib.dump(clf, 'naive_bayes_model.pkl')