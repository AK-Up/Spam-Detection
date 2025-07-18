from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
from collections import Counter

app = Flask(__name__)

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load your dataset (used to extract most common spam keywords)
# Replace with actual path and column names
df = pd.read_csv('spam.csv', encoding='latin-1')  # Example spam dataset
df = df[['v1', 'v2']]  # Assuming 'v1' is label and 'v2' is message
df.columns = ['label', 'message']

# Clean and extract most common spam words
spam_words = ' '.join(df[df['label'] == 'spam']['message']).lower()
spam_words = re.findall(r'\b[a-z]{3,}\b', spam_words)
top_keywords = [word for word, _ in Counter(spam_words).most_common(15)]

@app.route('/')
def home():
    return render_template('index.html', keywords=top_keywords)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = vectorizer.transform([message]).toarray()
    prediction = model.predict(data)
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template('index.html', message=message, prediction=result, keywords=top_keywords)

# if __name__ == '__main__':
#     app.run(debug=True)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)


