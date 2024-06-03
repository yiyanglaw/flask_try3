from flask import Flask, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re
import string

app = Flask(__name__)

# Load the spam detection model
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Sample data to test the model
data = pd.DataFrame({
    'Message': ['how are u?', 'Congratulations! You have won a free trip. Claim it now!'],
    'Spam': [0, 1]
})

# Training the model
clf.fit(data.Message, data.Spam)

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    text = request.form['text']
    processed_text = preprocess_text(text)
    prediction = clf.predict([processed_text])[0]
    if prediction == 0:
        result = 'Ham (Not Spam)'
    else:
        result = 'Spam'
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
