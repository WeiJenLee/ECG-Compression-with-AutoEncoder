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
    nor_lab = np.load('./dataset/test_name_Normal.npy')
    pvc_lab = np.load('./dataset/test_name_PVC.npy')
    nor_lab = np.reshape(nor_lab, (-1, 1))
    pvc_lab = np.reshape(pvc_lab, (-1, 1))
    label = np.concatenate([nor_lab, pvc_lab], axis=0)
    return dict(norm=norm, pvc=pvc, label=label)
