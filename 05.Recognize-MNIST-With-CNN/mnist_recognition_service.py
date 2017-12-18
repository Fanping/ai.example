import tensorflow as tf
import progressbar
from mnist_recognition_model import MnistRegModel

from tensorflow.examples.tutorials.mnist import input_data


class MnistRegService(object):
    def __init__(self):
        self.model = None

    def train(self, batch_size=64, epochs=10000):
        self.model = MnistRegModel(learning_rate=0.1)
        self.model.create()
        mnist = input_data.read_data_sets('../data/mnist', one_hot=True)

        variables = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(variables)
            saver = tf.train.Saver(max_to_keep=None)
            with progressbar.ProgressBar(max_value=epochs) as bar:
                for epoch in range(epochs):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    sess.run(
                        [self.model.optimizer, self.model.cross_entropy],
                        feed_dict={self.model.input: batch_xs,
                                   self.model.output: batch_ys,
                                   self.model.keep_prob: 0.5})
                    bar.update(epoch)
            saver.save(sess, 'model/' + self.model.name)

    def predict(self, img_arr):
        self.model = MnistRegModel(learning_rate=0.1)
        self.model.create()
        variables = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(variables)
            last_ckpt_path = tf.train.latest_checkpoint('model/')
            if last_ckpt_path is not None:
                saver = tf.train.Saver(max_to_keep=None)
                saver.restore(sess, last_ckpt_path)
            else:
                print('Not found the model.')
                return None
            ret = sess.run(self.model.net,
                           feed_dict={self.model.input: img_arr,
                                      self.model.keep_prob: 1.0})
            nums = []
            for r in ret:
                nums.append(sess.run(tf.argmax(r)))
        return nums

    def test(self):
        self.model = MnistRegModel(learning_rate=0.1)
        self.model.create()
        mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
        variables = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(variables)
            last_ckpt_path = tf.train.latest_checkpoint('model/')
            if last_ckpt_path is not None:
                saver = tf.train.Saver(max_to_keep=None)
                saver.restore(sess, last_ckpt_path)
            else:
                print('Not found the model.')
                return None
            print (
                sess.run(self.model.accuracy,
                         feed_dict={self.model.input: mnist.test.images,
                                    self.model.output: mnist.test.labels,
                                    self.model.keep_prob: 1.0}))
