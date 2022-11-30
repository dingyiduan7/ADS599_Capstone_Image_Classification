# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:16:17 2022

Reference:https://medium.com/@draj0718/deploying-deep-learning-model-using-flask-api-810047f090ac

"""

import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


app = Flask(__name__)
model = load_model('CNN_model.h5')
target_img = os.path.join(os.getcwd() , 'static/images')
@app.route('/')
def index_view():
    return render_template('index.html')
#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    img = cv2.imread(str(filename))
    x = cv2.resize(img, (224,224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = x.astype(np.float32)/255
    x = np.expand_dims(x, axis=0)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    return x
@app.route('/classify',methods=['GET','POST'])
def classify():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path) #prepressing method
            class_prediction=np.argmax(model.predict(img),axis=-1)
            if class_prediction == 0:
              result = "NORMAL"
            else:
              result = "PNEUMONIA"
            return render_template('classify.html', result = result, prob=class_prediction, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)