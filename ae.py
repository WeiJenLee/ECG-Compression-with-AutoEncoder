from __future__ import division
from utils import (nn, cnn, cnn_trans, GaussianSample,
                   kld, logpx, p_mse, cc, l1, mse)
import tensorflow as tf

z_dim = 14
input_dim = 70

class DAE:
    def __init__(self):
        self._encode = tf.make_template('Enc', self._encoder)
        self._decode = tf.make_template('Dec', self._decoder)

    def _encoder(self, x, training=False):
        with tf.name_scope('G'):
            #h1 = nn(x, 64, 'h1', training)
            #h2 = nn(x, 32, 'h2', training)
            outputs = tf.layers.dense(x, z_dim)
            #outputs = tf.layers.dropout(outputs, rate=0.1, training=training)
        return outputs

    def _decoder(self, x, training=False):
        with tf.name_scope('D'):
            #x = tf.reshape(x, shape=(-1, z_dim, 1, 1))
            #conv1 = cnn_trans(x, 4, [5, 1], [2, 1], 'conv1', training)
            #conv2 = cnn_trans(conv1, 8, [5, 1], [2, 1], 'conv2', training)
            #outputs = tf.contrib.layers.flatten(conv2)
            #h1 = nn(x, 64, 'h1', training)
            #h2 = nn(h1, 32, 'h2', training)
            outputs = tf.layers.dense(x, input_dim)
            #outputs = tf.layers.dropout(outputs, rate=0.1, training=training)
            #outputs = tf.nn.relu(outputs)
        return outputs

    def loss(self, x, training=True):
        with tf.name_scope('loss'):
            z = self._encode(x, training=training)
            x_h = self._decode(z, training=training)
            loss = dict()
            loss['pmse'] = p_mse(x, x_h)
            loss['corr'] = cc(x, x_h)
            #loss['diff'] = l1(x, x_h)
            tf.summary.scalar('pmse', loss['pmse'])
            tf.summary.scalar('corr', loss['corr'])
            #tf.summary.scalar('diff', loss['diff'])
        return loss

    def encode(self, x):
        return self._encode(x)

    def decode(self, x):
        return self._decode(x)


class VAE(DAE):
    def __init__(self):
        self._encode = tf.make_template('Enc', self._encoder)
        self._decode = tf.make_template('Dec', self._decoder)

    def _encoder(self, x, training=False):
        with tf.name_scope('G'):
            conv1 = cnn(x, 8, [5, 1], [2, 1], 'conv1', training)
            #conv2 = cnn(conv1, 8, [5, 1], [2, 1], 'conv2', training)
            outputs = tf.contrib.layers.flatten(conv1)
            #mean = tf.layers.dense(outputs, z_dim)
            #var = tf.layers.dense(outputs, z_dim)
            #h1 = nn(x, 32, 'h1', training)
            mean = tf.layers.dense(outputs, z_dim)
            var = tf.layers.dense(outputs, z_dim)
        return mean, var

    def loss(self, x, training=True):
        with tf.name_scope('loss'):
            x = tf.reshape(x, shape=(-1, input_dim, 1, 1))
            mean, var = self._encoder(x, training=training)
            z = GaussianSample(mean, var)
            x_h = self._decoder(z, training=training)
            loss = dict()
            loss['pmse'] = p_mse(x, x_h)
            loss['kld'] = tf.reduce_mean(
                kld(mean, var, tf.zeros_like(mean), tf.ones_like(var))
            )
            #loss['rec'] = tf.reduce_mean(
            #    logpx(x, xh, tf.zeros_like(xh))
            #)
            tf.summary.scalar('prd', loss['pmse'])
            tf.summary.scalar('kld', loss['kld'])
            #tf.summary.scalar('rec', loss['rec'])
        return loss

    def encode(self, x):
        x = tf.reshape(x, shape=(-1, input_dim, 1, 1))
        mean, _ =  self._encode(x)
        return mean
    
