import tensorflow as tf
import numpy as np
import os
import cv2
import glob
import math
from itertools import chain

import progressbar

from face_recognize_model import FaceRecognizeModel


class FaceRecognizeService(object):
    """service:train | test | predict"""

    def __init__(self):
        self.model = None

    def _convert_label_to_id(self, labels):
        att = []
        cnt = len(set(labels))
        for label in labels:
            val = np.zeros(cnt)
            val[label - 1] = 1
            att.append(val)
        return att

    def read_face_images(self, flag, file_dir, label_path):
        """
        :return:
        """
        images = []
        labels = []
        image_files = glob.glob(file_dir + "/*.bmp")
        cnt = len(image_files)
        for idx in range(cnt):
            image_file = os.path.join(file_dir,
                                      flag + "_" + str(idx + 1) + ".bmp")
            image = cv2.imread(image_file,
                               cv2.IMREAD_GRAYSCALE)
            image = (image / 255).tolist()
            image = list(chain(*image))
            images.append(image)

        label_file = open(label_path)
        lines = label_file.readlines()
        for line in lines:
            labels.append(int(line))

        labels = self._convert_label_to_id(labels)
        return images, labels

    def train(self, batch_size=64, epochs=100):
        self.model = FaceRecognizeModel()
        self.model.create()

        images, labels = self.read_face_images("train",
                                               "../data/face/train_image",
                                               "../data/face/train_label.txt")
        variables = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(variables)
            length = len(labels)
            times = math.ceil(length / batch_size)
            with progressbar.ProgressBar(max_value=epochs) as bar:
                for epoch in range(epochs):
                    for idx in range(times):
                        start = idx * batch_size
                        end = start + batch_size
                        batch_xs, batch_ys = images[start:end], \
                                             labels[start:end]
                        _, cross = sess.run(
                            [self.model.optimizer, self.model.cross_entropy],
                            feed_dict={self.model.input: batch_xs,
                                       self.model.output: batch_ys})
                    print('\n')
                    print(cross)
                    bar.update(epoch)
            saver = tf.train.Saver(max_to_keep=None)
            saver.save(sess, 'model/' + self.model.name)

    def test(self):
        self.model = FaceRecognizeModel()
        self.model.create()
        images, labels = self.read_face_images("test",
                                               "../data/face/test_image/",
                                               "../data/face/test_label.txt")

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
            acc = sess.run(self.model.accuracy,
                           feed_dict={self.model.input: images,
                                      self.model.output: labels})
            print("Accuracy on test: %f" % acc)
