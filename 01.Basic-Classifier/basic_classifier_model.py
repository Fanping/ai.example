import tensorflow as tf


class BasicClassifierModel(object):
    def __init__(self, in_size, out_size):
        self.name = "basic-classifier"
        self.input_size = in_size
        self.hiddens = [4, 8, 2]
        self.num_of_persons = out_size
        self.input = None
        self.output = None
        self.weights = None
        self.bias = None
        self.cost = None
        self.optimizer = None
        self.net = None
        self.accuracy = None

    def create(self, learning_rate):
        # 1. Init all
        self.input = tf.placeholder("float",
                                    [None, self.input_size])
        self.output = tf.placeholder("float",
                                     [None, self.num_of_persons])
        self.weights = {
            'h1': tf.Variable(
                tf.random_normal([self.input_size, self.hiddens[0]])),
            'h2': tf.Variable(
                tf.random_normal([self.hiddens[0], self.hiddens[1]])),
            'h3': tf.Variable(
                tf.random_normal([self.hiddens[1], self.hiddens[2]])),
            'out': tf.Variable(
                tf.random_normal([self.hiddens[2], self.num_of_persons]))
        }
        self.bias = {
            'h1': tf.Variable(tf.random_normal([self.hiddens[0]])),
            'h2': tf.Variable(tf.random_normal([self.hiddens[1]])),
            'h3': tf.Variable(tf.random_normal([self.hiddens[2]])),
            'out': tf.Variable(tf.random_normal([self.num_of_persons]))
        }

        # 2. Config layers.
        layer_1 = tf.add(tf.matmul(self.input, self.weights['h1']),
                         self.bias['h1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']),
                         self.bias['h2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_3 = tf.add(tf.matmul(layer_2, self.weights['h3']),
                         self.bias['h3'])
        layer_3 = tf.nn.relu(layer_3)
        output_layer = tf.add(tf.matmul(layer_3, self.weights['out']),
                              self.bias['out'])
        output_layer = tf.nn.softmax(output_layer)
        self.net = output_layer

        # 3. Define loss.
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,
                                                    labels=self.output))

        # 4. Define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost)

        # 5. Define accuracy
        correct_prediction = tf.equal(tf.argmax(output_layer, 1),
                                      tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
