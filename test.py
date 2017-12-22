import time, os, pickle
import math, random
import numpy as np
import tensorflow as tf
from os.path import join
from graph import ImportGraph
from scipy.io import savemat

def test(spec, path):
    norm = spec['norm']
    pvc = spec['pvc']
    label = spec['label']
    data = np.concatenate([norm, pvc], axis=0)
    #data = norm
    os.system("mkdir -p "+path)

    graph = ImportGraph(path)

    #for i in range(len(x)):
    prd = graph.prd(data)
    #cc = graph.cc(data)
    z = graph._encode(data)
    x_h = graph._decode(z)
    data = np.reshape(data, (-1, 350))
    x_h = np.reshape(x_h, (-1, 350))
        #print('Loss: %.4f' % loss)
        #count += loss
    #print('MSE: %.4f COR: %.4f' % loss['pmse'], loss['corr'])
    print(prd)
    #print(data[0], x_h[0])
    with open('label.txt', 'w') as f:
        for lab in label:
            f.write('%s\n' %lab)
    savemat(join(path, 'origin'), mdict={'ecg': data})
    savemat(join(path, 'recons'), mdict={'ecg': x_h})
