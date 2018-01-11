# -*- coding: utf-8 -*-
import tensorflow as tf
import glob
import os
import cv2
import string
import numpy as np
from CRNN_model import CRNN_Model
from PIL import Image


class CRNN_Service(object):
    def __init__(self):
        self.model = None
        self.chars = string.ascii_letters + string.digits

    def _label_to_array(self, label):
        return [self.chars.index(x) for x in label]

    def _convert_image(self, image_file):
        img = Image.open(image_file)
        img = img.convert('L')
        return np.asarray(img, dtype='float64') / 256.

    def _sparse_tuple_from(self, sequences, dtype=np.int32):
        indices = []
        values = []
        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), [i for i in range(len(seq))]))
            values.extend(seq)
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1],
                           dtype=np.int64)

        return indices, values, shape

    def _process_images(self, folder):
        image_files = glob.glob(folder + '/*.png')
        images = []
        labels = []
        for image_file in image_files:
            img = self._convert_image(
                image_file)
            img = img.reshape(img.shape[0], img.shape[1], 1)
            images.append(img)
            label = image_file.split("_")[-1].replace('.png', '')
            labels.append(self._label_to_array(label))
        return images, labels

    def train(self, num_epochs=10, batch_size=1):
        self.model = CRNN_Model()
        self.model.create(batch_size, len(self.chars))
        images, labels = self._process_images(
            '../data/identification_codes/train')
        variables = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(variables)
            saver = tf.train.Saver(max_to_keep=None)
            last_ckpt_path = tf.train.latest_checkpoint('model/')
            if last_ckpt_path is not None:
                saver.restore(session, last_ckpt_path)
            for epoch in range(num_epochs):
                print('Epoch:', epoch + 1)
                avg_loss = 0
                for idx in range(len(images)):
                    img = images[idx]
                    lab = self._sparse_tuple_from(np.asarray([labels[idx]]))
                    seq = np.reshape(np.array(4), (-1))
                    _, loss, _ = session.run(
                        [self.model.optimizer,
                         self.model.loss,
                         self.model.decode_ret],
                        feed_dict={
                            self.model.input: [img],
                            self.model.sequence_len: seq,
                            self.model.output: lab
                        })
                    avg_loss += loss
                print('loss:', avg_loss / len(images))
            saver.save(session, 'model/' + self.model.name)

    def predict(self, image_file):
        # lab=self._sparse_tuple_from(np.asarray([self._label_to_array('IBfQ')]))
        # ret = ''.join([self.chars[i] for i in lab[1]])
        # print(ret)
        self.model = CRNN_Model()
        self.model.create(1, len(self.chars))
        variables = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(variables)
            last_ckpt_path = tf.train.latest_checkpoint('model/')
            if last_ckpt_path is not None:
                saver = tf.train.Saver(max_to_keep=None)
                saver.restore(session, last_ckpt_path)
            else:
                print('Not found the model.')
                return None
            img = self._convert_image(
                image_file)
            img = img.reshape(img.shape[0], img.shape[1], 1)
            seq = np.reshape(np.array(4), (-1))
            feed_dict = {
                self.model.input: [img],
                self.model.sequence_len: seq,
            }
            decode_ret, decode_prob = session.run(
                [self.model.decode_ret, self.model.decode_prob], feed_dict)
            return ''.join(
                [self.chars[i] for i in decode_ret[0][1]])
