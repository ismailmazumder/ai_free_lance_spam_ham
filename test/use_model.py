import string
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# প্রিপ্রসেস ফাংশন
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# মডেল ও ভেক্টর লোড
tf = joblib.load('tfidf_vectorizer.pkl')
clf = joblib.load('naive_bayes_model.pkl')

# নতুন ইনপুট
your_sen = ["Don’t forget to bring the documents for the meeting."]
your_sen = [preprocess_text(sen) for sen in your_sen]
your_sen_vec = tf.transform(your_sen).toarray()

# প্রেডিকশন
prediction = clf.predict(your_sen_vec)
print("Prediction:", prediction[0])
