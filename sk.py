import  nltk
import json
import textblob
import langdetect
import os
import requests
import jwt
nltk.download('punkt')
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from textblob import TextBlob
from langdetect import detect
from bottle import route, run, request, response,Bottle
from os.path import join, dirname

app = Bottle()
vectorizer = TfidfVectorizer()
classifier = MultinomialNB()

@app.route('/test', method='OPTIONS')
def handle_options_request():
    # Set CORS headers for the preflight request
    response.headers['Access-Control-Allow-Origin'] = '*'  # Replace with your allowed domains
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
    return {}


    # Set CORS headers for POST request
   
    msg = data.get('agent_message')
    predicted_label = 'ignore'  # Replace with your prediction result
    return {'predicted_label': predicted_label}

def preprocess_text(text,language):
    with open('languages.json', 'r', encoding='utf-8') as file:
         language_data = json.load(file)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words(language_data[language]))
    #print(f"Three stop words for language {language}: {list(stop_words)[:3]}")
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(stemmed_tokens)
   

def check_message(msg):
    # Preprocess the single text
    new_text_preprocessed = preprocess_text(msg, detect(msg))
    # Transform the preprocessed text using the same vectorizer used for training
    new_text_transformed = vectorizer.transform([new_text_preprocessed])
    # Predict the label for the single text using the trained classifier
    predicted_label = classifier.predict(new_text_transformed)
    print(predicted_label[0])
    return predicted_label[0]
   


@app.route('/test', method='POST')
def initialize():
    response.headers['Access-Control-Allow-Origin'] = '*'  # Replace with your frontend origin
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type'
    data = request.json
    msg = data.get('agent_message')
    with open('train.json', 'r', encoding='utf-8') as file:
           data2 = json.load(file)
    texts = [item['text'] for item in data2]
    labels = [item['label'] for item in data2]     
    with open('languages.json', 'r', encoding='utf-8') as file:
         language_data = json.load(file)
    X_preprocessed = [preprocess_text(text,detect(text)) for text in texts]
    X_transformed = vectorizer.fit_transform(X_preprocessed)
    classifier.fit(X_transformed, labels)
    with open('testdata.json', 'r',encoding='utf-8') as file:
        test_data = json.load(file)
    X_test = [item['text'] for item in test_data]
    y_test = [item['label'] for item in test_data]
    X_test_preprocessed = [preprocess_text(text,detect(text)) for text in X_test]
    X_test_transformed = vectorizer.transform(X_test_preprocessed)
    y_pred = classifier.predict(X_test_transformed)
    print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    predicted_label = check_message(msg)
    return {'predicted_label': predicted_label}

    
    

app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))






