import os
import numpy as np
from glob import iglob
from scipy.io import loadmat, savemat
from os.path import join, basename, splitext

PERSON = ['105', '116', '215', '223']

def main(dirname):
    os.mkdir('dataset')
    for m in ['Normal', 'PVC']:
        train = np.zeros((1, 71))
        test = np.zeros((1, 71))
        for p in PERSON:
            path = join(dirname, m, p)
            data = np.zeros((1, 70))
            for f in iglob(path + '/*mat'):
                filename = join(path, f)
                print(f)
                mat = loadmat(filename)
                name = splitext(basename(f))[0]
                sig = mat['ecg'+name]
                sig = np.reshape(sig, (5, 70))
                data = np.concatenate([data, sig], axis=0)
            label = PERSON.index(p)*np.ones((len(data), 1))
            data = np.concatenate([data, label], axis=1)
            train = np.concatenate([train, data[len(data)//10:]], axis=0)
            test = np.concatenate([test, data[:len(data)//10]], axis=0)
        #np.random.shuffle(data)
        #_min = np.min(data, axis=1)
        #for i in range(len(data)):
        #    data[i] -= _min[i]
        #    data[i] *= 1000
        np.save('dataset/train_{}'.format(m), train[1:])
        np.save('dataset/test_{}'.format(m), test[1:])

if __name__ == '__main__':
    main('/mnt/gv0/user_wenlee/Data')
