import tensorflow as tf


class MnistRegModel(object):
    def __init__(self, learning_rate):
        self.name = 'mnist-reg-fcn'
        self.learning_rate = learning_rate
        self.input = tf.placeholder(tf.float32, [None, 784])
        self.output = tf.placeholder(tf.float32, [None, 10])
        self.net = None
        self.cross_entropy = None
        self.optimizer = None

    def create(self):
        # 1. Config layers.
        W1 = tf.Variable(tf.truncated_normal([784, 512], stddev=0.1))
        b1 = tf.Variable(tf.zeros([512]))
        W2 = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))
        b2 = tf.Variable(tf.zeros([10]))
        layer = tf.nn.relu(tf.matmul(self.input, W1) + b1)
        layer = tf.matmul(layer, W2) + b2
        self.net = layer
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.output,
                                                    logits=self.net))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.cross_entropy)
