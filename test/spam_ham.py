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

# ডেটা লোড ও ক্লিনিং
data = pd.read_csv("/home/ismail/PycharmProjects/free/test/data.csv")
data = data.drop_duplicates()
data['v2'] = data['v2'].fillna('')

# পাংচুয়েশন রিমুভ ফাংশন
def remove_punc(text):
    trans = str.maketrans('', '', string.punctuation)
    return text.translate(trans)

data['v2'] = data['v2'].apply(remove_punc)

# স্টপওয়ার্ড লোড করা
nltk.download('stopwords')
with open('bn_nlp/dataset/stop_word.txt', 'r') as sto:
    stops = set(sto.read().splitlines())

sw = set(stopwords.words('english'))
combined_stopwords = sw.union(stops)

def remove_sws(text):
    try:
        text = text.lower()
    except:
        pass
    return " ".join([word for word in text.split() if word not in combined_stopwords])

data['v2'] = data['v2'].apply(remove_sws)

# লেমাটাইজেশন
nltk.download('wordnet')
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()

def lemma(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

data['v2'] = data['v2'].apply(lemma)

# ফিচার ভেক্টর তৈরি
tf = TfidfVectorizer(max_features=3000)
x = tf.fit_transform(data['v2']).toarray()
y = data['v1']

# ট্রেন টেস্ট ভাগ
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# মডেল ট্রেন
clf = MultinomialNB()
clf.fit(X_train, y_train)

# টেস্টিং ও রিপোর্ট
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# মডেল ও ভেক্টরাইজার সেভ
joblib.dump(tf, 'tfidf_vectorizer.pkl')
joblib.dump(clf, 'naive_bayes_model.pkl')
