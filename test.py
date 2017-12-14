import time, os, pickle
import math, random
import numpy as np
import tensorflow as tf
from graph import ImportGraph
from scipy.io import savemat

def test(spec, path):
    norm = spec['norm']
    pvc = spec['pvc']
    data = np.concatenate([norm[:, ::-1], pvc[:, :-1]], axis=0)
    #data = norm
    os.system("mkdir -p "+path)

    graph = ImportGraph(path)

    #for i in range(len(x)):
    loss = graph.loss(data)
    z = graph._encode(data)
    x_h = graph._decode(z)
    data = np.reshape(data, (-1, 350))
    x_h = np.reshape(x_h, (-1, 350))
        #print('Loss: %.4f' % loss)
        #count += loss
    #print('MSE: %.4f COR: %.4f' % loss['pmse'], loss['corr'])
    print(loss)
    #print(data[0], x_h[0])
    label = np.concatenate([norm[:, -1], pvc[:, -1]], axis=0)
    data = np.concatenate([data, label], axis=1)
    x_h = np.concatenate([x_h, label], axis=1)
    savemat('origin', mdict={'ecg': data})
    savemat('recons', mdict={'ecg': x_h})
