from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = vectorizer.transform([message]).toarray()  # Transform the input message using the vectorizer
    prediction = model.predict(data)

    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template('index.html', message=message, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
