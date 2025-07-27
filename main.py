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
    words = word_tokenize(translated.text)
    filtered = [word for word in words if word.lower() not in stopwords.words('english') and word.isalpha()]
    print("Raw English Keywords:\n", ' '.join(filtered))

