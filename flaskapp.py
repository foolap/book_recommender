from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from main import recommend, similar
from flask_wtf import FlaskForm
from wtforms import StringField
import requests

app = Flask(__name__)

prediction = recommend(similar,'To Kill a Mockingbird')
BACKEND_URL = 'your_backend_url'



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['book_input']
        recommended_books_list = recommend(similar,user_input)
        return render_template('index.html', user_input=user_input, recommended_books=recommended_books_list)
    return render_template('index.html', user_input=None, recommended_books=None)

from main import selected_books
#print(selected_books['Book-Title'])
class BookForm(FlaskForm):
    book_input = StringField('Book')


def autocomplete():
    form = BookForm()
    query = request.form.get('query')

    # Replace this with your actual autocomplete logic
    # For now, let's return some dummy suggestions
    suggestions = ['Autocomplete 1', 'Autocomplete 2', 'Autocomplete 3']

    return jsonify({'suggestions': suggestions})




if __name__ == '__main__':
    app.run(debug=True)