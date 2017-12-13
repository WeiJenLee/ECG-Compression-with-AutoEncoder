import time, os, pickle
import math, random
import numpy as np
import tensorflow as tf
from graph import ImportGraph
from scipy.io import savemat

def test(spec, path):
    norm = spec['norm']
    pvc = spec['pvc']
    data = np.concatenate([norm, pvc], axis=0)
    #data = norm
    os.system("mkdir -p "+path)

    graph = ImportGraph(path)

    #for i in range(len(x)):
    loss = graph.loss(data)
    z = graph._encode(data)
    x_h = graph._decode(z)
        #print('Loss: %.4f' % loss)
        #count += loss
    #print('MSE: %.4f COR: %.4f' % loss['pmse'], loss['corr'])
    print(loss)
    #print(data[0], x_h[0])
    savemat('origin', mdict={'ecg': data})
    savemat('recons', mdict={'ecg': x_h})
