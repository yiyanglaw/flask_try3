from flask import Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

app = Flask(__name__)

# Load the data for training
data = pd.read_csv('sms_3.csv')

# Fill missing values with an empty string
data['Message'] = data['Message'].fillna('')
data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Preprocess the text
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

data['Message'] = data['Message'].apply(preprocess_text)

# Create a vocabulary
vocab = set()
for text in data['Message']:
    vocab.update(text.split())

vocab_to_index = {word: index for index, word in enumerate(vocab)}
index_to_vocab = {index: word for word, index in vocab_to_index.items()}

# Tokenize and encode the text
def encode_text(text):
    tokens = text.split()
    encoded = [vocab_to_index[token] for token in tokens if token in vocab_to_index]
    return encoded

X = data['Message'].apply(encode_text)
y = data['Spam'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create PyTorch datasets and data loaders
train_data = TensorDataset(torch.tensor([x for x in X_train]), torch.tensor(y_train))
test_data = TensorDataset(torch.tensor([x for x in X_test]), torch.tensor(y_test))
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the deep learning model
class SpamClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SpamClassifier, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embeddings = self.embeddings(x)
        _, (hidden, _) = self.lstm(embeddings)
        hidden = hidden.squeeze(0)
        output = self.fc(hidden)
        return output

# Instantiate the model
vocab_size = len(vocab_to_index)
embedding_dim = 128
hidden_dim = 256
output_dim = 1
model = SpamClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_data)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.6f}')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')

# Define a function to predict spam
def predict_spam(text):
    encoded_text = encode_text(text)
    if not encoded_text:
        return "Ham (Not Spam)"
    tensor_text = torch.tensor(encoded_text).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor_text)
        prediction = (output.squeeze() > 0).item()
    if prediction:
        return "Spam"
    else:
        return "Ham (Not Spam)"

# Flask route
@app.route('/predict_spam', methods=['POST'])
def predict_spam_route():
    text = request.form['text']
    result = predict_spam(text)
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000,debug=False)
