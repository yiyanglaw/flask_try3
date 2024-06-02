from flask import Flask, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import string

app = Flask(__name__)

model = load_model('spam_detection_model.h5')

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    text = request.form['text']
    processed_text = preprocess_text(text)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([processed_text])
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100, padding='post')
    prediction = model.predict(padded_sequence)[0][0]
    if prediction > 0.5:
        result = 'Spam'
    else:
        result = 'Ham'
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
