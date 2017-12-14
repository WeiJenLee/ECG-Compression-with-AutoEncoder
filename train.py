import time, os, pickle
import math, random
import numpy as np
import tensorflow as tf
from ae import(z_dim, input_dim,
               DAE, VAE)

iter_num = 50000
batch_num = 64


def train(data, path):
    os.system("mkdir -p " + path)
    norm = data['norm']
    pvc = data['pvc']
    feat = np.concatenate([norm[:, :-1], pvc[:, :-1]], axis=0)
    #feat = norm
    x = tf.placeholder(tf.float32, [None, input_dim], name='x')
    z = tf.placeholder(tf.float32, [None, z_dim], name='z')

    model = DAE()
    loss = model.loss(x)
    _z = model.encode(x)
    x_h = model.decode(z)
    tf.add_to_collection('encode', _z)
    tf.add_to_collection('decode', x_h)
    tf.add_to_collection('loss', loss['pmse'])
    w_loss = loss['pmse'] - loss['corr']

    optimize = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.9, beta2=0.99).minimize(w_loss)
    #optimize = tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(loss)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        start = time.time()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(path+'logs', sess.graph)

        for i in range(iter_num):
            np.random.shuffle(feat)
            for j in range(0, len(feat), batch_num):
                _ = sess.run([optimize], feed_dict={x:feat[j:j+batch_num]})
            if i%100 == 0:
                xb = feat[j:]
                _, err, result = sess.run([optimize,loss, merged],
                                          feed_dict={x:xb})
                writer.add_summary(result, len(feat)*i+j)
                print('Epoch [%4d] Time [%5.4f] PRD: [%.4f] CC: [%.4f]'%
                  (i+1, time.time() - start, err['pmse'], err['corr']))
            if i%1000 == 0:
                saver.save(sess, path+'test_best_model')

