import tensorflow as tf


class TextClassifierRNNModel(object):
    def __init__(self,
                 seq_length=600,
                 num_classes=10,
                 vocab_size=5000,
                 embedding_dim=64,
                 learning_rate=1e-3):
        self.name = 'text_classify_rnn'
        # 1. Define input & output.
        self.input = tf.placeholder(tf.int32, [None, seq_length],
                                    name='input')
        self.output = tf.placeholder(tf.float32,
                                     [None, num_classes],
                                     name='output')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 2. Define embedding.
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size,
                                                      embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input)

        # 3. Define RNN.
        with tf.name_scope("rnn"):
            cells = [self._create_cell() for _ in range(2)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                            inputs=embedding_inputs,
                                            dtype=tf.float32)
            last = _outputs[:, -1, :]

        # 4. Define FCN.
        with tf.name_scope("fcn"):
            fc_1 = tf.layers.dense(last, 128,
                                   name='fc_1')
            fc_1 = tf.contrib.layers.create_cell(fc_1, self.keep_prob)
            fc_1 = tf.nn.relu(fc_1)
            self.net = tf.layers.dense(fc_1, num_classes,
                                       name='fc_2')
            self.predict = tf.argmax(tf.nn.softmax(self.net), 1)

        # 5. Define optimization.
        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.net, labels=self.output)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.loss)

        # 6. Define accuracy.
        with tf.name_scope("accuracy"):
            correct_predict = tf.equal(tf.argmax(self.output, 1), self.predict)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    def _lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(128,
                                            state_is_tuple=True)

    def _gru_cell(self):
        return tf.contrib.rnn.GRUCell(128)

    def _create_cell(self):
        cell = self.gru_cell()
        return tf.contrib.rnn.DropoutWrapper(cell,
                                             output_keep_prob=self.keep_prob)
