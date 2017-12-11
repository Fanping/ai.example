import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import progressbar

from slim_basic_classifier_model import SlimBasicClassifierModel


class SlimBasicClassifierService(object):
    def __init__(self, input_size, output_size, learning_rate=0.001):
        self.model = SlimBasicClassifierModel(input_size, output_size)
        self.model.create(learning_rate)

    def train(self, train_x, train_y, epochs=1000):
        variables = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(variables)
            saver = tf.train.Saver(max_to_keep=None)
            with progressbar.ProgressBar(max_value=epochs) as bar:
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
                    avg_cost = avg_cost / length
                    plt.plot(epoch, avg_cost, 'bo')
                    bar.update(epoch)
            plt.title(self.model.name + ' training line')
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.savefig('epoch.png', dpi=200)
            print('Epoch:', '%04d' % (epoch + 1), 'final cost=',
                  '{:.9f}'.format(avg_cost))
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
