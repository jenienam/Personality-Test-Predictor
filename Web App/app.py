# Dependencies
import numpy as np
from flask import Flask, request, render_template
import requests
from web_scraper import commentScraper
from data_cleaning import calculateModelParameters

app = Flask(__name__)

@app.route('/')
def home():
    return (render_template('index.html'))

@app.route('/result',methods = ['POST', 'GET'])
def predict():
    if request.method == 'POST':
        username = request.form['username']

        comments = commentScraper(username)

        scores = calculateModelParameters(username)
        
        return (render_template('result.html', username=username, comments=comments, scores=scores))


# @app.route('/result',methods = ['POST', 'GET'])
# def get_input():
#     if request.method == 'POST':
#         username = request.form['username']
#         return (render_template('result.html', username=username))

if __name__ == "__main__":
    app.run(debug=True)