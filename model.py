from __future__ import division
from utils import (nn, cnn, cnn_trans, GaussianSample, cross_entropy,
                   _compute_gradient_penalty, GaussianLogDensity,
                   kld, logpx, p_mse, cc, l1, mse)
import tensorflow as tf

z_dim = 7
input_dim = 70

class DAE:
    def __init__(self):
        self._encode = tf.make_template('Enc', self._encoder)
        self._decode = tf.make_template('Dec', self._decoder)

    def _encoder(self, x, training=False):
        with tf.name_scope('G'):
            x = tf.reshape(x, shape=(-1, input_dim, 1, 1))
            conv1 = cnn(x, 8, [5, 1], [1, 1], 'conv1', training)
            conv2 = cnn(conv1, 16, [5, 1], [1, 1], 'conv2', training)
            conv3 = cnn(conv2, 8, [5, 1], [1, 1], 'conv3', training)
            conv3 = conv1 + conv3
            outputs = tf.contrib.layers.flatten(conv3)
            #h1 = nn(x, 64, 'h1', training)
            #h2 = nn(h1, 32, 'h2', training)
            outputs = tf.layers.dense(outputs, z_dim)
        return outputs

    def _decoder(self, x, training=False):
        with tf.name_scope('D'):
            x = tf.reshape(x, shape=(-1, z_dim, 1, 1))
            conv1 = cnn_trans(x, 8, [5, 1], [1, 1], 'conv1', training)
            conv2 = cnn_trans(conv1, 16, [5, 1], [1, 1], 'conv2', training)
            conv3 = cnn_trans(conv2, 8, [5, 1], [1, 1], 'conv3', training)
            conv3 - conv1 + conv3
            outputs = tf.contrib.layers.flatten(conv3)
            #h1 = nn(x, 64, 'h1', training)
            #h2 = nn(h1, 32, 'h2', training)
            #outputs = nn(h2, 16, 'h3', training)
            outputs = tf.layers.dense(outputs, input_dim)
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


#class VAE(DAE):
#    def __init__(self):
#        self._encode = tf.make_template('Enc', self._encoder)
#        self._decode = tf.make_template('Dec', self._decoder)
#
#    def _encoder(self, x, training=False):
#        with tf.name_scope('G'):
#            #x = tf.reshape(x, shape=(-1, input_dim, 1, 1))
#            #conv1 = cnn(x, 4, [5, 1], [2, 1], 'conv1', training)
#            #conv2 = cnn(x, 8, [5, 1], [2, 1], 'conv2', training)
#            #outputs = tf.contrib.layers.flatten(conv2)
#            #mean = tf.layers.dense(outputs, z_dim)
#            #var = tf.layers.dense(outputs, z_dim)
#            h1 = nn(x, 64, 'h1', training)
#            outputs = nn(x, 32, 'h2', training)
#            mean = tf.layers.dense(outputs, z_dim)
#            var = tf.layers.dense(outputs, z_dim)
#        return mean, var
#
#    def loss(self, x, training=True):
#        with tf.name_scope('loss'):
#            mean, var = self._encode(x, training=training)
#            z = GaussianSample(mean, var)
#            x_h = self._decode(z, training=training)
#            loss = dict()
#            loss['pmse'] = p_mse(x, x_h)
#            #loss['corr'] = z_dim*cc(x, x_h)
#            loss['kld'] = tf.reduce_mean(
#                kld(mean, var)
#            )
#            loss['logpx'] = tf.reduce_mean(
#                logpx(x, x_h, tf.zeros_like(x_h))
#            )
#            tf.summary.scalar('kld', loss['kld'])
#            tf.summary.scalar('prd', loss['pmse'])
#            #tf.summary.scalar('corr', loss['corr'])
#            #tf.summary.scalar('rec', loss['rec'])
#        return loss
#
#    def encode(self, x):
#        mean, _ =  self._encode(x)
#        return mean
#
#class AAE(VAE):    
#    def __init__(self):
#        self._encode = tf.make_template('Enc', self._encoder)
#        self._decode = tf.make_template('Dec', self._decoder)
#        self._discriminate = tf.make_template('Dis', self._discriminator)
#
#    def _discriminator(self, x, training=False):
#        with tf.name_scope('D'):
#            conv1 = cnn(x, 8, [2, 1], [2, 1], 'conv1', training)
#            conv2 = cnn(conv1, 16, [2, 1], [2, 1], 'conv2', training)
#            conv2 = tf.contrib.layers.flatten(conv2)
#            h1 = nn(conv2, 128, 'h1', training)
#            outputs = tf.layers.dense(h1, 1)
#        return outputs
#
#    def loss(self, x, y):
#        with tf.name_scope('loss'):
#            #latent sample
#            z_mu, z_lv = self._encode(x)
#            z = GaussianSampleLayer(z_mu, z_lv)
#            #real sample
#            sample = GaussianSampleLayer(tf.zeros_like(z_mu),tf.ones_like(z_lv))
#
#            e = tf.random_uniform([tf.shape(x)[0], 1], 0., 1., name='epsilon')
#            r_intp = sample + e * (z - sample)
#            z_all = tf.concat([sample, z, r_intp], axis=0)
#
#            _j = self._discriminate(z_all)
#            j_real, j_fake, j_intp = tf.split(_j, 3)
#
#            xh = self._decode(z)
#             
#            gp = _compute_gradient_penalty(j_intp, r_intp)
#            mean = -1. * tf.reduce_mean(
#                GaussianLogDensity(
#                    x=j_real,
#                    mu=tf.zeros_like(j_real),
#                    log_var=tf.ones_like(j_real),
#                )
#            )
#
#            #logPx = tf.reduce_mean(
#            #    GaussianLogDensity(
#            #        slim.flatten(x),
#            #        slim.flatten(xh),
#            #        tf.zeros_like(slim.flatten(xh))),
#            #)
#
#            #D_KL = tf.reduce_mean(
#            #    GaussianKLD(
#            #        slim.flatten(z_mu),
#            #        slim.flatten(z_lv),
#            #        slim.flatten(tf.zeros_like(z_mu)),
#            #        slim.flatten(tf.ones_like(z_lv)),
#            #    )
#            #)
#
#        loss = dict()
#        loss['D_real'] = tf.reduce_mean(j_real)
#        loss['D_fake'] = tf.reduce_mean(j_fake)
#        loss['D_dist'] = loss['D_real'] - loss['D_fake']
#        loss['l_D'] = - loss['D_dist'] + gp + mean
#        loss['G'] = - logPx
#        loss['l_E'] = -loss['D_fake']
#        loss['logP'] = logPx
#        #loss['KLD'] = D_KL
#
#        tf.summary.scalar('l_D', loss['l_D'])
#        tf.summary.scalar('l_E', loss['l_E'])
#        tf.summary.scalar('logPx', logPx)
#        tf.summary.scalar('KLD', D_KL)
#
#        tf.summary.histogram('xh', xh)
#        tf.summary.histogram('x', x)
#        tf.summary.histogram('z', z)
#        return loss
