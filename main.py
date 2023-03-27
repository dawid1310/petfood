import numpy as np
from utils import gen_id
from threading import Thread
import sys
import os
import json
import os
from tensorflow.keras.models import load_model
from utils import resize_im

# global settings
prediction_folder = 'predictions'
images_folder = 'images'
model_file = '_model_.hdf5'
input_size = 224

# globals
breeds = None
model = None

def read_breeds():
    f = open('rasy_lista.txt', 'r')
    lines = f.readlines()
    global breeds
    breeds = []
    for line in lines:
        breeds.append(line.strip())
    f.close()
    breeds.sort()

def load_nn_model():
    print('LOADING MODEL')
    global model
    model = load_model(model_file)
    print('DONE')

def predict(im, num, save_im):
    print(im)
    id_ = gen_id()
    thr = Thread(target=_predict, args=(im, num, id_, save_im))
    thr.start()
    return id_

def _predict(im, num, id, save_im=False):
    im = resize_im(im, input_size, True)
    im = np.expand_dims(im, axis=0)
    probs = model.predict(im).flatten()
    probs /= probs.sum()
    probs *= 100
    indxs = probs.argsort()[::-1]
    _perc = []
    _breeds = []
    sum = 0.0
    for i in range(num):
        _breeds.append(breeds[indxs[i]])
        _perc.append(str(probs[indxs[i]]))
        sum += probs[indxs[i]]
    _breeds.append('Other')
    _perc.append(str(100.0 - sum))
    preds = {'id': id, 'breeds': _breeds, 'percent': _perc}
    print(preds)
    with open(os.path.join(prediction_folder, id+'.json'),'w') as fp:
        json.dump(preds, fp)
    if save_im:
        im.save(os.path.join(images_folder, id+'.png'))

def _predict_random_test(im, num, id, save_im=False):
    probs = np.random.rand(len(breeds))
    probs[np.random.randint(len(breeds))] = 955
    probs[np.random.randint(len(breeds))] = 455
    probs /= probs.sum()
    probs *= 100
    indxs = probs.argsort()[::-1]
    _perc = []
    _breeds = []
    sum = 0.0
    for i in range(num):
        _breeds.append(breeds[indxs[i]])
        _perc.append(probs[indxs[i]])
        sum += _perc[-1]
    _breeds.append('Other')
    _perc.append(100.0 - sum)
    preds = {'id': id, 'breeds': _breeds, 'percent': _perc}
    print(preds)
    with open(os.path.join(prediction_folder, id+'.json'),'w') as fp:
        json.dump(preds, fp)
    if save_im:
        im.save(os.path.join(images_folder, id+'.png'))

def get_status(id):
    path = os.path.join(prediction_folder,id+'.json')
    if not os.path.exists(path):
        return 'not-ready'
    return 'ready'

def get_breeds_stats(id):
    if get_status(id) == 'ready':
        path = os.path.join(prediction_folder,id+'.json')
        with open(path, 'r') as fp:
            preds = json.load(fp)
        return preds
    return None

