sudo apt update
sudo apt upgrade -y
sudo apt install python3,python3-pip -y
pip install pytesseract,langdetect,nltk,pandas
pip install scikit-learn
pip install banglanltk
pip install googletrans==4.0.0-rc1
sudo apt install tesseract-ocr -y
sudo apt install libtesseract-dev -y
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
git clone https://github.com/tesseract-ocr/tessdata_best.git
sudo mv tessdata_best/* /usr/share/tesseract-ocr/5/tessdata/