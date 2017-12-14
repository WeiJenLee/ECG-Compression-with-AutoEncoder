import os
import numpy as np
from glob import iglob
from scipy.io import loadmat, savemat
from os.path import join, basename, splitext

def main(dirname):
    os.mkdir('dataset')
    for m in ['Normal', 'PVC']:
        path = join(dirname, m)
        train = np.zeros((1, 70))
        test = np.zeros((1, 70))
        for f in iglob(path + '/*mat'):
            filename = join(path, f)
            print(f)
            mat = loadmat(filename)
            name = splitext(basename(f))[0]
            sig = mat['ecg'+name]
            sig = np.reshape(sig, (5, 70))
            train = np.concatenate([train, sig[:5]], axis=0)
            test = np.concatenate([test, sig[5]], axis=0)
        train = train[1:]
        test = test[1:]
        #np.random.shuffle(data)
        #_min = np.min(data, axis=1)
        #for i in range(len(data)):
        #    data[i] -= _min[i]
        #    data[i] *= 1000
        np.save('dataset/train_{}'.format(m), train)
        np.save('dataset/test_{}'.format(m), test)

if __name__ == '__main__':
    main('/mnt/gv0/user_wenlee/Data')
