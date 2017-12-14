import os
import numpy as np
from glob import iglob
from scipy.io import loadmat, savemat
from os.path import join, basename, splitext

TYPE = ['Normal', 'PVC']
PERSON = ['105', '116', '215', '223']

def main(dirname):
    os.mkdir('dataset')
    for m in TYPE:
        train = np.zeros((1, 70))
        test = np.zeros((1, 70))
        train_lab = []
        test_lab = []
        for p in PERSON:
            label = []
            path = join(dirname, m, p)
            data = np.zeros((1, 70))
            for f in iglob(path + '/*mat'):
                filename = join(path, f)
                print(f)
                mat = loadmat(filename)
                name = splitext(basename(f))[0]
                label.append(name+m)
                sig = mat['ecg'+name]
                sig = np.reshape(sig, (5, 70))
                data = np.concatenate([data, sig], axis=0)
            train = np.concatenate([train, data[1:-25]], axis=0)
            train_lab.append(np.asarray(label[:-5]))
            test = np.concatenate([test, data[-25:]], axis=0)
            test_lab.append(np.asarray(label[-5:]))
        #np.random.shuffle(data)
        #_min = np.min(data, axis=1)
        #for i in range(len(data)):
        #    data[i] -= _min[i]
        #    data[i] *= 1000
        np.save('dataset/train_{}'.format(m), train[1:])
        np.save('dataset/test_{}'.format(m), test[1:])
        np.save('dataset/train_name_{}'.format(m), train_lab)
        np.save('dataset/test_name_{}'.format(m), test_lab)

if __name__ == '__main__':
    main('/mnt/gv0/user_wenlee/Data')
