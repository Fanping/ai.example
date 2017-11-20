import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from basic_classifier_model import BasicClassifierModel


class BasicClassifierService(object):
    def __init__(self, input_size, output_size, learning_rate=0.001):
        self.model = BasicClassifierModel(input_size, output_size)
        self.model.create(learning_rate)

    def train(self, train_x, train_y, epochs=1000):
        variables = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(variables)
            saver = tf.train.Saver(max_to_keep=None)
            t_index = 0
            for epoch in range(epochs):
                avg_cost = 0.0
                length = len(train_x)
                for index in range(length):
                    input_data = np.asarray([[train_x[index]]])
                    label_data = np.asarray([train_y[index]])
                    _, c = sess.run([self.model.optimizer, self.model.cost],
                                    feed_dict={
                                        self.model.input: input_data,
                                        self.model.output: label_data})
                    avg_cost += c
                    t_index += 1
                avg_cost = avg_cost / length
                # plt.plot(t_index, avg_cost, 'cost')
                print('Epoch:', '%04d' % (epoch + 1), 'cost=',
                      '{:.9f}'.format(avg_cost))
            # plt.xlabel('Epoch')
            # plt.ylabel('Cost')
            # plt.savefig('epoch.png', dpi=200)
            saver.save(sess, 'model/' + self.model.name)

    def test(self, test_x, test_y):
        variables = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(variables)
            saver = tf.train.Saver(max_to_keep=None)
            last_ckpt_path = tf.train.latest_checkpoint('model/')
            if last_ckpt_path is not None:
                saver.restore(sess, last_ckpt_path)
            else:
                print('Not found the model.')
                return None
            return self.model.accuracy.eval({self.model.input: np.asarray(
                [[test_x]]),
                self.model.output: np.asarray([test_y])})

    def predict(self, data_x):
        variables = tf.initialize_all_variables()
        actual_results = []
        with tf.Session() as sess:
            sess.run(variables)
            saver = tf.train.Saver(max_to_keep=None)
            last_ckpt_path = tf.train.latest_checkpoint('model/')
            if last_ckpt_path is not None:
                saver.restore(sess, last_ckpt_path)
            else:
                print('Not found the model.')
                return None
            for x in data_x:
                y = sess.run([self.model.net],
                             feed_dict={self.model.input: np.asarray(
                                 [[x]])})
                actual_results.append(y)
        return actual_results
