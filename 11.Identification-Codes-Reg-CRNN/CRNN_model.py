import tensorflow as tf
import string
from tensorflow.contrib import rnn


class CRNN_Model(object):
    def __init__(self):
        self.name = 'crnn-model'
        self.input = None
        self.output = None
        self.sequence_len = None
        self.net = None
        self.num_classes = -1
        self.loss = None
        self.optimizer = None
        self.decode_ret = None
        self.decode_prob = None

    def _create_rnn(self, inputs):
        """
        Create Bidirectionnal LSTM Recurrent Neural Network.
        """
        lstm_forward_cell = rnn.BasicLSTMCell(256, forget_bias=1.0)
        lstm_backward_cell = rnn.BasicLSTMCell(256, forget_bias=1.0)
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_forward_cell,
                                                     lstm_backward_cell,
                                                     inputs, dtype=tf.float32)
        return outputs

    def _create_cnn(self, inputs):
        """
        Create Convolutionnal Neural Network.
        """

        # 64 / 3 x 3 / 1 / 1
        conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(3, 3),
                                 padding="same", activation=tf.nn.relu)

        # 2 x 2 / 1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                        strides=2)

        # 128 / 3 x 3 / 1 / 1
        conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                                 padding="same", activation=tf.nn.relu)

        # 2 x 2 / 1
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                        strides=2)

        # 256 / 3 x 3 / 1 / 1
        conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                                 padding="same", activation=tf.nn.relu)

        # 256 / 3 x 3 / 1 / 1
        conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=(3, 3),
                                 padding="same", activation=tf.nn.relu)

        # 1 x 2 / 1
        pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[1, 2],
                                        strides=2)

        # 512 / 3 x 3 / 1 / 1
        conv5 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                                 padding="same", activation=tf.nn.relu)

        # Batch normalization layer
        bnorm1 = tf.layers.batch_normalization(conv5)

        # 512 / 3 x 3 / 1 / 1
        conv6 = tf.layers.conv2d(inputs=bnorm1, filters=512, kernel_size=(2, 2),
                                 padding="same", activation=tf.nn.relu)

        # Batch normalization layer
        bnorm2 = tf.layers.batch_normalization(conv6)

        # 1 x 2 / 2
        pool4 = tf.layers.max_pooling2d(inputs=bnorm2, pool_size=[2, 2],
                                        strides=2)

        # 512 / 2 x 2 / 1 / 0
        conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(2, 2),
                                 padding="valid", activation=tf.nn.relu)

        return conv7

    def _feature_2_sequences(self, feature):
        return tf.unstack(tf.squeeze(feature, [1]))

    def create(self, batch_size, num_classes):
        """
        Create model.
        """
        self.input = tf.placeholder(tf.float32,
                                    [batch_size, 40, 80, 1], name='input')
        self.output = tf.sparse_placeholder(tf.int32, name='output')
        self.sequence_len = tf.placeholder(tf.int32, [None],
                                           name='sequence_len')
        self.num_classes = num_classes + 2

        # 1. CNN model.
        cnn = self._create_cnn(self.input)

        # 2. RNN model.
        rnn = self._create_rnn(self._feature_2_sequences(cnn))

        # 3. FCN model.
        W = tf.Variable(
            tf.truncated_normal([512, self.num_classes], stddev=0.1),
            name="W")
        b = tf.Variable(tf.constant(0., shape=[self.num_classes]), name="b")
        fcn_layer = tf.matmul(tf.reshape(rnn, [-1, 512]), W) + b
        logits = tf.reshape(fcn_layer, [batch_size, -1,
                                        self.num_classes])
        self.net = tf.transpose(logits, (1, 0, 2))

        # 4. Loss and optimizer.
        self.loss = tf.reduce_mean(
            tf.nn.ctc_loss(self.output, self.net, self.sequence_len))

        self.optimizer = tf.train.MomentumOptimizer(0.1, 0.9).minimize(
            self.loss)

        # 5. Decode result.
        self.decode_ret, self.decode_prob = tf.nn.ctc_beam_search_decoder(
            self.net,
            self.sequence_len)
