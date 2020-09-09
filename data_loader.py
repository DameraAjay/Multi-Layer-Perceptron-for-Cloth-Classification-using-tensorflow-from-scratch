import os
import sys
import gzip
import numpy as np

def load_data(type):
    lbls_path = os.path.join('./data/fashion','%s-labels-idx1-ubyte.gz'% type)
    imgs_path = os.path.join('./data/fashion','%s-images-idx3-ubyte.gz'% type)
    with gzip.open(lbls_path, 'rb') as lpath:
        lbls = np.frombuffer(lpath.read(), dtype=np.uint8,offset=8)
    with gzip.open(imgs_path, 'rb') as imgpath:
        imgs = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(lbls), 784)
    return imgs,lbls