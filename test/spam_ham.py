import pandas as pd
data = pd.read_csv("/home/ismail/PycharmProjects/free/test/spam_final.csv")
# null
data = data.drop_duplicates()
data.duplicated().sum()
from nltk.tokenize import word_tokenize
#import nltk
import banglanltk as nltk
from banglanltk import word_tokenize
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkkt_tab')

data['v2'] = data['v2'].fillna('') # nan(not a number) also kind of NONE type thing. ekhane amra oi value gula change kore dibo
data['v2'].apply(lambda x: word_tokenize(x))
import string
def remove_punc(text):
  trans = str.maketrans('', '', string.punctuation)
  return text.translate(trans)

data['v2'] = data['v2'].apply(remove_punc)
data['v2']

import nltk
from bn_nlp.preprocessing import ban_processing

nltk.download('stopwords')
from bn_nlp.preprocessing import ban_processing

bp = ban_processing()
from nltk.corpus import stopwords

# bangla
with open('bn_nlp/dataset/stop_word.txt', 'r') as sto:
  # stops=sto.read()
  stops = set(sto.read().splitlines())

sw = set(stopwords.words('english'))
combined_stopwords = sw.union(stops)


def remove_sws(text):
  # low char
  try:
    text = text.lower()
  except Exception as e:
    pass
  non_stop_word = []
  split_text = text.split()
  for new in split_text:
    if new not in combined_stopwords:
      non_stop_word.append(new)
      # from nltk.stem import WordNetLemmatizer
      # lemmatizer = WordNetLemmatizer()
      # new = lemmatizer.lemmatize(new)
      # non_stop_word.append(new)
  return " ".join(non_stop_word)


# def remove_sws(text):
#   s = [word.lower() for word in text.split() if word.lower() not in sw]
#   return " ".join(s)
data['v2'] = data['v2'].apply(remove_sws)
data['v2']

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def lemma(text):
  lemmatize_Word = []
  split_text = text.split()
  for new in split_text:
    if new not in sw:
      lemmatize_Word.append(new)
      from nltk.stem import WordNetLemmatizer
      lemmatizer = WordNetLemmatizer()
      new = lemmatizer.lemmatize(new)
      lemmatize_Word.append(new)
  return " ".join(lemmatize_Word)


def lemma(text):
  l = [lemmatizer.lemmatize(word) for word in text.split()]
  return " ".join(l)


data['v2'] = data['v2'].apply(lemma)
data['v2']
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features=3000)
x = tf.fit_transform(data['v2']).toarray()
y = data['v1']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x)
from sklearn.naive_bayes import  MultinomialNB
# Create the instance of Naive Bayes
clf = MultinomialNB()
# Fit the data
clf.fit(X_train, y_train)
# Making prediction



y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))


def preprocess_text(text):
  stop_words = set(stopwords.words('english'))
  text = text.lower()  # Lowercase
  text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
  text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
  return text


# tokeize korte hobe
your_sen = ["Dear user, your account has been flagged for suspicious activity. Please verify your identity immediately to avoid deactivation: http://verify-now.bad-link.net"]

# Preprocess the new sentence before vectorizing
your_sen = [preprocess_text(sen) for sen in your_sen]

# Transform the new sentence using the same vectorizer
your_sen_transformed = tf.transform(your_sen).toarray()
# data['v2'].apply(lambda x: word_tokenize(x))
# Predict the label for the new sentence
your_sen_pred = clf.predict(your_sen_transformed)

# Print the prediction for the new sentence
print("Prediction for your sentence:")
print(your_sen_pred[0], "chuppppp")