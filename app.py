from flask import Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.models import Sequential
import re
import string

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('sms_3.csv')

# Preprocess the category labels
data['Category'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Data preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        # Remove punctuation and convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        # Remove non-word characters
        text = re.sub(r'\W+', ' ', text)
        return text
    else:
        return ''  # Return empty string for NaN values

data['Message'] = data['Message'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Category'], test_size=0.2, random_state=42)

# Tokenize and pad the sequences
vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=100)
vectorizer.adapt(X_train.values)

# Define the model architecture
model = Sequential([
    vectorizer,
    Embedding(len(vectorizer.get_vocabulary()), 100, input_length=100),
    Conv1D(128, 5, padding='valid', activation='relu', strides=1),
    MaxPooling1D(),
    Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
    GlobalAveragePooling1D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Define text preprocessing function for prediction
def preprocess_text_for_prediction(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    text = request.form['text']
    processed_text = preprocess_text_for_prediction(text)
    sequence = vectorizer([processed_text])
    prediction = model.predict(sequence)[0][0]
    if prediction > 0.5:
        result = 'Spam'
    else:
        result = 'Ham'
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
