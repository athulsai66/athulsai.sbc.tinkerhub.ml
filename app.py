from flask import Flask, render_template, request
import sys
import os
from numpy.core.records import array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
sys.path.append(os.path.abspath('./model'))
from keras.models import load_model


app = Flask(__name__)
model = load_model('model.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def prediction():
    imagefile= request.files['imagefile']
    classification = ['','']
    classification[0]= "images/"+ imagefile.filename
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)

    image1 = load_img(image_path, target_size=(224, 224))##loading the image
    
    
    image1 = np.asarray(image1) ##converting to an array
    image1 = image1 / 255 ##scaling by doing a division of 255
    plt.imshow(image1)
    image1 = np.expand_dims(image1, axis=0) ##expanding the dimensions
    output = model.predict(image1)
    if ((output[0][0]>0.8000) or (output[0][1]>0.8000)):
        if output[0][0] > output[0][1]:
            classification[1] = "Image is of a chakka"##chakka
        else:
            classification[1] = "Image is of a manga"##manga
    else:
        if output[0][0] > output[0][1]:
            classification[1] = "Sorry I can't recognise this image but it can be " + str(round((output[0][0]/1)*100)) + "% chakka"##chakka
        else:
            classification[1] = "Sorry I can't recognise this image but it can be " + str(round((output[0][1]/1)*100)) + "% manga"##manga
    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)