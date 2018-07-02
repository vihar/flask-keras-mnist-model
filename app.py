from flask import Flask, render_template
import keras
import tensorflow as tf
import pandas as pd


app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=8000)
