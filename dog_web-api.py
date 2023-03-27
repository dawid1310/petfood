from flask import Flask
from flask import request, jsonify
import random
import numpy as np
from main import get_status
from main import predict
from main import get_breeds_stats
from main import read_breeds
from main import load_nn_model
import warnings
import cv2

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/api/predict', methods=['GET'])
def do_predict():
    if 'image' not in request.args:
        return jsonify({'error': 'No image provided.'}), 400
    try:
        im = cv2.imread(request.args['image'])
        id = predict(im, int(request.args['num']), request.args['save_im'])
        return id
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    if 'id' not in request.args:
        return jsonify({'error': 'No ID provided.'}), 400
    try:
        id = request.args['id']
        return get_status(id)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/breeds_stats', methods=['GET'])
def breeds_stats():
    if 'id' not in request.args:
        return jsonify({'error': 'No ID provided.'}), 400
    try:
        id = request.args['id']
        return get_breeds_stats(id)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    read_breeds()
    load_nn_model()
    warnings.filterwarnings('ignore')
    app.run(host='0.0.0.0',port=5000)
