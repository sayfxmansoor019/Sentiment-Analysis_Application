# app.py
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the trained model
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        data_tfidf = tfidf.transform(data)
        prediction = model.predict(data_tfidf)
        sentiment = ''
        if prediction == 1:
            sentiment = 'Positive'
        elif prediction == 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return render_template('index.html', prediction_text=f'Sentiment: {sentiment}', message=message)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([data['message']])
    sentiment = ''
    if prediction == 0:
        sentiment = 'Negative'
    elif prediction == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Neutral'
    return jsonify(sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)