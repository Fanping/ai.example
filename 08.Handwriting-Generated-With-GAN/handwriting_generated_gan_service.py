import tensorflow as tf
import progressbar
import numpy as np
import os
from skimage.io import imsave
from handwriting_generated_gan_model import HandwritingGANModel

from tensorflow.examples.tutorials.mnist import input_data


class HandwritingGANService(object):
    def __init__(self):
        self.model = None

    def train(self, batch_size=256, epochs=10000):
        self.model = HandwritingGANModel(batch_size=batch_size,
                                         learning_rate=0.0001)
        mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
        sample_val = np.random.normal(0, 1, size=(batch_size, 100)).astype(
            np.float32)
        variables = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(variables)
            saver = tf.train.Saver(max_to_keep=None)
            steps = 60000 / batch_size
            with progressbar.ProgressBar(max_value=epochs) as bar:
                for epoch in range(epochs):
                    for idx in np.arange(steps):
                        batch_xs, _ = mnist.train.next_batch(batch_size)
                        batch_xs = 2 * batch_xs.astype(np.float32) - 1
                        z_value = np.random.normal(0, 1, size=(
                            batch_size, 100)).astype(
                            np.float32)
                        sess.run(self.model.discriminator_trainer,
                                 feed_dict={self.model.input_image: batch_xs,
                                            self.model.input_prior: z_value,
                                            self.model.keep_prob: np.sum(
                                                0.7).astype(
                                                np.float32)})
                        if idx % 1 == 0:
                            sess.run(self.model.generator_trainer,
                                     feed_dict={
                                         self.model.input_image: batch_xs,
                                         self.model.input_prior: z_value,
                                         self.model.keep_prob: np.sum(
                                             0.7).astype(
                                             np.float32)})
                    gen_val = sess.run(self.model.generator,
                                       feed_dict={
                                           self.model.input_prior: sample_val})
                    self._save(gen_val, "output/epoch-{0}.jpg".format(epoch))
                    bar.update(epoch)
            saver.save(sess, 'model/' + self.model.name)

    def _save(self, batch_res, fname, grid_size=(8, 8), grid_pad=5):
        if not os.path.exists('output'):
            os.makedirs('output')
        batch_res = 0.5 * batch_res.reshape(
            (batch_res.shape[0], 28, 28)) + 0.5
        img_h, img_w = batch_res.shape[1], batch_res.shape[2]
        grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
        grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
        img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        for i, res in enumerate(batch_res):
            if i >= grid_size[0] * grid_size[1]:
                break
            img = (res) * 255
            img = img.astype(np.uint8)
            row = (i // grid_size[0]) * (img_h + grid_pad)
            col = (i % grid_size[1]) * (img_w + grid_pad)
            img_grid[row:row + img_h, col:col + img_w] = img
        imsave(fname, img_grid)
