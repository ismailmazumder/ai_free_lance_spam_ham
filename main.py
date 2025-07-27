from PIL import Image
import pytesseract
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
translator = Translator()
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
def remove_stop_words(text, stop_words):
    words = text.split()
    filtered_text = ' '.join(word for word in words if word.lower() not in stop_words)
    return filtered_text

if __name__ == "__main__":
    # Example usage
    # take the image path as argument

    image_path = 'mixed_final.png'  # Replace with your image path
    texts = pytesseract.image_to_string(Image.open(image_path), lang='eng+hin')
    translated = translator.translate(texts, dest='en')
    # print(translated.text)
    # words = word_tokenize(translated.text)
    # filtered = [word for word in words if word.lower() not in stopwords.words('english') and word.isalpha()]
    # print("Raw English Keywords:\n", ' '.join(filtered))
    # check by lsdir if any pkl file here or not
    import os

    # বর্তমান ডিরেক্টরিতে থাকা সব ফাইলের লিস্ট
    files = os.listdir(".")

    # শুধু .pkl ফাইল ফিল্টার করে দেখানো
    pkl_files = [file for file in files if file.endswith('.pkl')]

    if pkl_files:
        print("Found .pkl files:")
        for file in pkl_files:
            print(file)
    else:
        print("No .pkl files found in this directory.")
        import subprocess
        subprocess.run(["python3", "spam_ham.py"])

    from use_model import *
    from use_model import preprocess_text
    your_sen = [" ".join(translated.text)]
    your_sen = [preprocess_text(sen) for sen in your_sen]
    your_sen_vec = tf.transform(your_sen).toarray()

    # প্রেডিকশন
    prediction = clf.predict(your_sen_vec)
    print("Prediction:", prediction[0])



