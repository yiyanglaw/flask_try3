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

# Load the trained model
model = load_model('spam_detection_model.h5')

# Preprocess text for prediction
def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    # Remove non-word characters
    text = re.sub(r'\W+', ' ', text)
    
    return text

@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    # Get user input from the request
    text = request.form['text']
    
    # Preprocess the user input
    processed_text = preprocess_text(text)
    
    # Tokenize and pad the sequence
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)  # Assuming same vocab size
    tokenizer.fit_on_texts([processed_text])
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100, padding='post')
    
    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]
    
    # Return the prediction
    return str(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
