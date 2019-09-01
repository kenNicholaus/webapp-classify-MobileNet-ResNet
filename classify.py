from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


# Keras
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import preprocess_input
#from keras.models import load_model
from keras.models import Model, load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
model = keras.applications.mobilenet.MobileNet()
model1 = ResNet50(weights='imagenet')
print('Models loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.mobilenet.preprocess_input(x)

    preds = model.predict(x)
    return preds

def model_predict1(img_path, model1):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    y = image.img_to_array(img)
    #y = np.true_divide(y, 255)
    y = np.expand_dims(y, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    y = keras.applications.imagenet_utils.preprocess_input(y, mode='caffe' )

    preds1 = model.predict(y)
    return preds1





@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        preds1 = model_predict1(file_path, model1)

        # Process your result
        pred_class = decode_predictions(preds, top=1)
        pred_class1 = decode_predictions(preds1, top=1)
        result = ('Mobilenet :'+str(pred_class[0][0][1])+', prob: '+str((pred_class[0][0][2]).round(3))+'%'+'  '+' ResNet :'+str(pred_class1[0][0][1])+', prob: '+str((pred_class1[0][0][2]).round(3))+'%')

        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
