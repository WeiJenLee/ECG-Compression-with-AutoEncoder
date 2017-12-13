import os
import numpy as np
from glob import iglob
from scipy.io import loadmat, savemat
from os.path import join, basename, splitext

def main(dirname):
    os.mkdir('dataset')
    for m in ['Normal', 'PVC']:
        path = join(dirname, m)
        data = np.zeros((1, 70))
        for f in iglob(path + '/*mat'):
            filename = join(path, f)
            print(f)
            mat = loadmat(filename)
            name = splitext(basename(f))[0]
            sig = mat['ecg'+name]
            sig = np.reshape(sig, (5, 70))
            data = np.concatenate([data, sig], axis=0)
        data = data[1:]
        np.random.shuffle(data)
        _min = np.min(data, axis=1)
        for i in range(len(data)):
            data[i] -= _min[i]
            data[i] *= 1000
        np.save('dataset/train_{}'.format(m), data[:-50])
        np.save('dataset/test_{}'.format(m), data[-50:])

if __name__ == '__main__':
    main('/mnt/gv0/user_wenlee/Data')
