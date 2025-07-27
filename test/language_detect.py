from googletrans import Translator
translator = Translator()


text = "Hello my name is  राहुल and I live in दिल्ली. Today I went fdfkdjfd to the market and खरीदा some सब्जियाँ and fruits. It was a good day but बहुत थक गया हूँ।"

# Auto-detect source language
translated = translator.translate(text, dest='en')  # Translate to Bangla

print("Translated Text:\n", translated.text)
