import tensorflow as tf


class FaceRecognizeModel(object):
    def __init__(self):
        self.name = 'face-reg'
        self.input = tf.placeholder(tf.float32, [None, 40 * 55])
        self.output = tf.placeholder(tf.float32, shape=[None, 100])
        self.cross_entropy = None
        self.net = None
        self.optimizer = None
        self.correct_prediction = None
        self.accuracy = None

    def _weight_variable(self, shape):
        return tf.get_variable("weight", shape,
                               initializer=tf.random_normal_initializer())

    def _bias_variable(self, shape):
        return tf.get_variable("bias", shape,
                               initializer=tf.random_normal_initializer())

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

    def create(self, learning_rate=1e-3):
        # 1. Create first layer.
        image = tf.reshape(self.input, [-1, 55, 40, 1])

        with tf.variable_scope("conv_layer_1") as conv_layer_1:
            conv1_w = self._weight_variable([3, 3, 1, 32])
            conv1_b = self._bias_variable([32])
            conv1_layer = tf.nn.relu(
                tf.add(self._conv2d(image, conv1_w), conv1_b))
            conv1_max_pool = self._max_pool(conv1_layer)

        # 2. The second layer of convolution.
        with tf.variable_scope("conv_layer_2") as conv_layer_2:
            conv2_w = self._weight_variable([3, 3, 32, 64])
            conv2_b = self._bias_variable([64])
            conv2_layer = tf.nn.relu(
                self._conv2d(conv1_max_pool, conv2_w) + conv2_b)
            conv2_max_pool = self._max_pool(conv2_layer)
            conv2_flatten = tf.contrib.layers.flatten(conv2_max_pool)

        # 3. The first full connection layer.
        with tf.variable_scope("full_layer_1") as full_layer_1:
            fc1_w = self._weight_variable([10 * 14 * 64, 1024])
            fc1_b = self._bias_variable([1024])
            fc1_layer = tf.nn.relu(
                tf.add(tf.matmul(conv2_flatten, fc1_w), fc1_b))

        # 4. The second full connection layer.
        with tf.variable_scope("full_layer_2") as full_layer_2:
            fc2_w = self._weight_variable([1024, 100])
            fc2_b = self._bias_variable([100])
            fc2_layer = tf.add(tf.matmul(fc1_layer, fc2_w), fc2_b)

        self.net = fc2_layer

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.output,
                                                    logits=self.net))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate).minimize(
            self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.net, 1),
                                           tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, "float"))
