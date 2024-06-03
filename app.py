from flask import Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.callbacks import EarlyStopping
import re
import string

app = Flask(__name__)

# Load the data for training
data = pd.read_csv('sms_3.csv')

# Fill missing values with an empty string
data['Message'] = data['Message'].fillna('')

# Encode labels
label_encoder = LabelEncoder()
data['Spam'] = label_encoder.fit_transform(data['Category'])

X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25)

# Preprocessing function
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# GridSearch parameters
params = {
    'tfidf__max_features': [1000, 2000, 3000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'clf__epochs': [10, 20, 30],
    'clf__batch_size': [32, 64, 128]
}

# Create a pipeline with TF-IDF vectorizer and LSTM classifier
clf = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=preprocess_text)),
    ('clf', Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=100),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ]))
])

# GridSearchCV
grid_search = GridSearchCV(clf, params, cv=3, n_jobs=-1)

# Training the model
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Evaluation
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    text = request.form['text']
    processed_text = preprocess_text(text)
    prediction = grid_search.predict([processed_text])[0]
    if prediction == 0:
        result = 'Ham (Not Spam)'
    else:
        result = 'Spam'
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000,debug=False)
