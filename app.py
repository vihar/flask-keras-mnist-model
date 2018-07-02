from flask import Flask, render_template,request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys 
import os
sys.path.append(os.path.abspath("./model"))
from load import * 


global graph, model

model, graph = init()

app = Flask(__name__)


@app.route('/')
def index_view():
    return render_template('index.html')

def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
		output.write(imgstr.decode('base64'))

@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData)

	x = imread('output.png',mode='L')

	x = np.invert(x)

	x = imresize(x,(28,28))


	x = x.reshape(1,28,28,1)

	with graph.as_default():
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))

		response = np.array_str(np.argmax(out,axis=1))
		return response	

if __name__ == '__main__':
    app.run(debug=True, port=8000)
