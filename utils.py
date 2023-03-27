import datetime
import os
import pickle
import copy
import cv2
import numpy as np

def gen_id():
    return str(datetime.datetime.now()).replace(':', '.').replace(' ','-')

def get_status(id):
    path = os.path.join('models',id)
    if not os.path.exists(path):
        return 'no-model'
    ff = open(os.path.join(path,'status'), 'r')
    stat = ff.readline().strip()
    ff.close()
    return stat

def resize_im(im, s, loop=False):
    if im.shape[0] < im.shape[1]: #horizontal image
        p = int(s*im.shape[0]/im.shape[1])
        im = cv2.resize(im, (s, p))
        if loop:
            im = copy.deepcopy(cv2.vconcat([im, copy.deepcopy(im[:(s-p),:,:])]))
        else:
            im = cv2.vconcat([im, np.zeros((s-int(s*im.shape[0]/im.shape[1]), s, 3), dtype=np.uint8)])
    else:
        p = int(s*im.shape[1]/im.shape[0])
        im = cv2.resize(im, (p,s))
        if loop:
            im = copy.deepcopy(cv2.hconcat([im, copy.deepcopy(im[:,:(s-p),:])]))
        else:
            im = cv2.hconcat([im, np.zeros((s, s-p, 3), dtype=np.uint8)])
    if im.shape[0] != s or im.shape[1] != s:
        im = cv2.resize(im, (s,s))
    return im