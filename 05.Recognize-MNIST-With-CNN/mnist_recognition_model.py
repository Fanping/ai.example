import tensorflow as tf


class MnistRegModel(object):
    def __init__(self, learning_rate):
        self.name = 'mnist-reg-cnn'
        self.input = tf.placeholder(tf.float32, [None, 28 * 28])
        self.output = tf.placeholder(tf.float32, shape=[None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.net = None
        self.cross_entropy = None
        self.optimizer = None
        self.correct_prediction = None
        self.accuracy = None
        self.learning_rate = learning_rate

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv2d(self, input_feature, filter):
        return tf.nn.conv2d(input_feature,
                            filter,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

    def _max_pool(self, feature_map):
        return tf.nn.max_pool(feature_map,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    def create(self):
        image = tf.reshape(self.input, [-1, 28, 28, 1])
        # 1. The first layer of convolution.
        conv1_w = self._weight_variable([5, 5, 1, 32])
        conv1_b = self._bias_variable([32])
        conv1_layer = tf.nn.relu(self._conv2d(image, conv1_w) + conv1_b)
        conv1_max_pool = self._max_pool(conv1_layer)

        # 2. The second layer of convolution.
        conv2_w = self._weight_variable([5, 5, 32, 64])
        conv2_b = self._bias_variable([64])
        conv2_layer = tf.nn.relu(
            self._conv2d(conv1_max_pool, conv2_w) + conv2_b)
        conv2_max_pool = self._max_pool(conv2_layer)
        conv2_flatten = tf.reshape(conv2_max_pool, [-1, 7 * 7 * 64])

        # 3. The first full connection layer.
        fc1_w = self._weight_variable([7 * 7 * 64, 1024])
        fc1_b = self._bias_variable([1024])
        fc1_layer = tf.nn.relu(tf.matmul(conv2_flatten, fc1_w) + fc1_b)
        fc1_layer_dropout = tf.nn.dropout(fc1_layer, self.keep_prob)

        # 4. The second full connection layer.
        fc2_w = self._weight_variable([1024, 10])
        fc2_b = self._bias_variable([10])
        self.net = tf.nn.softmax(tf.matmul(fc1_layer_dropout, fc2_w) + fc2_b)
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.output,
                                                    logits=self.net))
        self.optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(
            self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.net, 1),
                                           tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, "float"))
