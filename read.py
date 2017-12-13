from __future__ import print_function
import numpy as np
import pickle

def load_train_data():
    norm = np.load('./dataset/train_Normal.npy')
    pvc = np.load('./dataset/train_PVC.npy')
    return dict(norm=norm, pvc=pvc)            

def load_test_data():
    norm = np.load('./dataset/test_Normal.npy')
    pvc = np.load('./dataset/test_PVC.npy')
    return dict(norm=norm, pvc=pvc)            
