import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers


class SlimBasicClassifierModel(object):
    def __init__(self, in_size, out_size):
        self.name = "slim-basic-classifier"
        self.input_size = in_size
        self.output_size = out_size
        self.input = None
        self.output = None
        self.cost = None
        self.optimizer = None
        self.net = None
        self.accuracy = None

    def create(self, learning_rate):
        # 1. Init all
        self.input = tf.placeholder("float",
                                    [None, self.input_size])
        self.output = tf.placeholder("float",
                                     [None, self.output_size])

        # 2. Define network.
        self.net = slim.fully_connected(self.input, 4,
                                        scope='fc/fc_1')
        self.net = slim.fully_connected(self.net, 8,
                                        scope='fc/fc_2')
        self.net = slim.fully_connected(self.net, self.output_size,
                                        scope='fc/fc_3')
        # 3. Define loss.
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.net,
                                                    labels=self.output))

        # 4. Define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost)

        # 5. Define accuracy
        correct_prediction = tf.equal(tf.argmax(self.net, 1),
                                      tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
