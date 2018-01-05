import tensorflow as tf


class FaceGANModel(object):
    def __init__(self, batch_size=64, learning_rate=1e-3):
        # 1. Define input.
        self.input_image = tf.placeholder(tf.float32, [batch_size, 40 * 55],
                                          name="input_image")
        self.input_prior = tf.placeholder(tf.float32, [batch_size, 100],
                                          name="input_prior")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # 2. Define generator.
        generator_w1 = tf.Variable(tf.truncated_normal([100, 150], stddev=0.1),
                                   name="g_w1", dtype=tf.float32)
        generator_b1 = tf.Variable(tf.zeros([150]), name="g_b1",
                                   dtype=tf.float32)
        generator_layer1 = tf.nn.relu(
            tf.matmul(self.input_prior, generator_w1) + generator_b1)
        generator_w2 = tf.Variable(tf.truncated_normal([150, 300], stddev=0.1),
                                   name="g_w2", dtype=tf.float32)
        generator_b2 = tf.Variable(tf.zeros([300]), name="g_b2",
                                   dtype=tf.float32)
        generator_layer2 = tf.nn.relu(
            tf.matmul(generator_layer1, generator_w2) + generator_b2)
        generator_w3 = tf.Variable(
            tf.truncated_normal([300, 40 * 55], stddev=0.1),
            name="g_w3", dtype=tf.float32)
        generator_b3 = tf.Variable(tf.zeros([40 * 55]), name="g_b3",
                                   dtype=tf.float32)
        generator_layer3 = tf.matmul(generator_layer2,
                                     generator_w3) + generator_b3
        self.generator = tf.nn.tanh(generator_layer3)

        # 3. Define discriminator.
        x_in = tf.concat([self.input_image, self.generator], 0)
        discriminator_w1 = tf.Variable(
            tf.truncated_normal([40 * 55, 300], stddev=0.1),
            name="d_w1", dtype=tf.float32)
        discriminator_b1 = tf.Variable(tf.zeros([300]), name="d_b1",
                                       dtype=tf.float32)
        discriminator_layer1 = tf.nn.dropout(
            tf.nn.relu(tf.matmul(x_in, discriminator_w1) + discriminator_b1),
            self.keep_prob)
        discriminator_w2 = tf.Variable(
            tf.truncated_normal([300, 150], stddev=0.1),
            name="d_w2", dtype=tf.float32)
        discriminator_b2 = tf.Variable(tf.zeros([150]), name="d_b2",
                                       dtype=tf.float32)
        discriminator_layer2 = tf.nn.dropout(tf.nn.relu(
            tf.matmul(discriminator_layer1,
                      discriminator_w2) + discriminator_b2), self.keep_prob)
        discriminator_w3 = tf.Variable(
            tf.truncated_normal([150, 1], stddev=0.1),
            name="d_w3",
            dtype=tf.float32)
        discriminator_b3 = tf.Variable(tf.zeros([1]), name="d_b3",
                                       dtype=tf.float32)
        discriminator_h3 = tf.matmul(discriminator_layer2,
                                     discriminator_w3) + discriminator_b3
        y_data = tf.nn.sigmoid(
            tf.slice(discriminator_h3, [0, 0], [batch_size, -1]))
        self.discriminator = tf.nn.sigmoid(
            tf.slice(discriminator_h3, [batch_size, 0], [-1, -1]))

        # 4.Define loss
        discriminator_loss = - (tf.log(y_data) + tf.log(1 - self.discriminator))
        generator_loss = - tf.log(self.discriminator)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.discriminator_trainer = self.optimizer.minimize(discriminator_loss,
                                                             var_list=[
                                                                 discriminator_w1,
                                                                 discriminator_b1,
                                                                 discriminator_w2,
                                                                 discriminator_b2,
                                                                 discriminator_w3,
                                                                 discriminator_b3])
        self.generator_trainer = self.optimizer.minimize(generator_loss,
                                                         var_list=[generator_w1,
                                                                   generator_b1,
                                                                   generator_w2,
                                                                   generator_b2,
                                                                   generator_w3,
                                                                   generator_b3])
