# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from cnews_loader import *
from text_classify_rnn_model import TextClassifierRNNModel


class TextClassifierRNNService(object):
    def __init__(self):
        self.model = None
        self.base_dir = '../data/text-classification'
        self.train_file = os.path.join(self.base_dir, 'cnews.train.txt')
        self.test_file = os.path.join(self.base_dir, 'cnews.test.txt')
        self.val_file = os.path.join(self.base_dir, 'cnews.val.txt')
        self.vocab_file = os.path.join(self.base_dir, 'cnews.vocab.txt')
        self.seq_length = 600
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(self.vocab_file)
        self.model = TextClassifierRNNModel()

    def train(self, num_epochs=10, batch_size=128):
        x_train, y_train = process_file(self.train_file,
                                        self.word_to_id,
                                        self.cat_to_id,
                                        self.seq_length)

        variables = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(variables)
            saver = tf.train.Saver(max_to_keep=None)
            total_batch = 0
            last_acc = 0
            flag = False
            for epoch in range(num_epochs):
                print('Epoch:', epoch + 1)
                batch_train = get_batches(x_train, y_train, batch_size)
                for x_batch, y_batch in batch_train:
                    feed_dict = {
                        self.model.input: x_batch,
                        self.model.output: y_batch,
                        self.model.keep_prob: 0.5
                    }
                    if total_batch % 100 == 0:
                        feed_dict[self.model.keep_prob] = 1.0
                        loss_train, acc_train = session.run(
                            [self.model.loss, self.model.accuracy],
                            feed_dict=feed_dict)
                        print (acc_train)
                        if abs(acc_train - last_acc) < 0.01:
                            saver.save(sess=session, save_path="./model")
                            flag = True
                            break
                        last_acc = acc_train
                    session.run(self.model.optimizer, feed_dict=feed_dict)
                    total_batch += 1
                if flag:
                    break

    def test(self):
        x_test, y_test = process_file(self.test_file,
                                      self.word_to_id,
                                      self.cat_to_id,
                                      self.seq_length)

        variables = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(variables)
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(sess=session, save_path='./model')
            data_len = len(x_test)
            batch_eval = get_batches(x_test, y_test, 128)
            total_loss = 0.0
            total_acc = 0.0
            for x_batch, y_batch in batch_eval:
                batch_len = len(x_batch)
                feed_dict = {
                    self.model.input: x_batch,
                    self.model.output: y_batch,
                    self.model.keep_prob: 1
                }
                loss, acc = session.run([self.model.loss, self.model.accuracy],
                                        feed_dict=feed_dict)
                total_loss += loss * batch_len
                total_acc += acc * batch_len

            loss_test = total_loss / data_len
            acc_test = total_acc / data_len
            msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
            print(msg.format(loss_test, acc_test))

    def predict(self, content):
        data_id = get_content_ids(content,
                                  self.word_to_id,
                                  self.seq_length)
        variables = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(variables)
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(sess=session, save_path='./model')
            feed_dict = {
                self.model.input: data_id,
                self.model.keep_prob: 1.0
            }
            return self.categories[
                session.run(self.model.predict, feed_dict=feed_dict)[0]]
