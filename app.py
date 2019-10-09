from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
from keras.preprocessing import image
import numpy as np
import pandas as pd
from ml_model.run_pipeline import train_model
from ml_model.prediction import make_prediction
from sqlalchemy import create_engine
from keras.models import model_from_json, load_model

from keras import backend as K
# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
PORT = os.environ['PORT']# os.environ['PORT']
UPLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/uploaded_images'
WEIGHTS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/ml_model/saved_model/weigths.h5'
ARCHITECTURE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/ml_model/saved_model/architecture.json'
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/ml_model/saved_model/best_model.h5'
UPLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/uploaded_images'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Init db
db = SQLAlchemy(app)
# Init ma
ma = Marshmallow(app)


# make a predict
@app.route('/predict', methods=['POST'])
def predict():
    K.clear_session()

    """"# Model reconstruction from JSON file
    with open(ARCHITECTURE_PATH, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(WEIGHTS_PATH)"""
    model = load_model(MODEL_PATH)
    imagefile = request.files.get('image', '')

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'imagefile')
    imagefile.save(file_path)

    test_image = image.load_img(file_path, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] == 0:
        pred = 'hotdog'
    else:
        pred = 'not hotdog'
    os.remove(file_path)
    K.clear_session()
    return jsonify(pred)


@app.route('/train', methods=['GET'])
def train():
    train_model()
    return "Training completed"


@app.route('/status', methods=['GET'])
def status():
    return "Everythign gucci"


# Run Server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)
