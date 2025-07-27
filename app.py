import os
import pandas as pd
import subprocess
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from googletrans import Translator

from use_model import tf, clf, preprocess_text

UPLOAD_FOLDER = 'static/uploads'
DATA_PATH = 'data.csv'
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

translator = Translator()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_text = ''

    if request.method == 'POST':
        # টেক্সট সাবমিশন
        if 'text_input' in request.form and request.form['text_input'].strip():
            input_text = request.form['text_input']
        elif 'image_file' in request.files:
            file = request.files['image_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                raw = pytesseract.image_to_string(Image.open(path), lang='eng+hin')
                trans = translator.translate(raw, dest='en')
                input_text = trans.text
            else:
                flash('Invalid image file')
                return redirect(request.url)

        if input_text:
            proc = preprocess_text(input_text)
            vec = tf.transform([proc]).toarray()
            pred = clf.predict(vec)[0]
            prediction = pred

    return render_template('index.html', prediction=prediction, input_text=input_text)

@app.route('/modify', methods=['POST'])
def modify():
    text = request.form['input_text']
    label = request.form['correct_label']

    # Save to data.csv
    df = pd.read_csv(DATA_PATH)
    df = pd.concat([df, pd.DataFrame([[label, text]], columns=['v1', 'v2'])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    # Retrain
    subprocess.run(['python3', 'spam_ham.py'], check=True)

    return "Training complete!"

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
