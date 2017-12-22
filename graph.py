import tensorflow as tf

class ImportGraph():
    def __init__(self, path):
        self.path = path
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.path+'test_best_model.meta', clear_devices=True)
            saver.restore(self.sess, self.path+'test_best_model')
            self.encode=tf.get_collection('encode')[0]
            self.decode=tf.get_collection('decode')[0]
            self.pmse = tf.get_collection('prd')
            self.corr = tf.get_collection('cc')

    def _encode(self, data):
        return self.sess.run(self.encode, feed_dict={'x:0': data})

    def _decode(self, data):
        return self.sess.run(self.decode, feed_dict={'z:0': data})

    def prd(self, x):
        return self.sess.run(self.pmse, feed_dict={'x:0': x})

    def cc(self, x):
        return self.sess.run(self.corr, feed_dict={'x:0': x})


