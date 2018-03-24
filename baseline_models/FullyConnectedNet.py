import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/', one_hot=True)

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def plot_digits(*args):
    # indices of the image of a certain digt\it
    digit_indices = np.array(list(args))
    num_indices = len(digit_indices)
    fig, axes = plt.subplots(ncols=num_indices, figsize=(num_indices, 1))
    for i in range(num_indices):
        ax = axes.flatten()[i]
        img = mnist.train.images[digit_indices[i]].reshape(28, 28)
        label = np.argmax(mnist.train.labels[digit_indices[i]])
        ax.set_title('Label: {}'.format(label))
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.show()

class FCN():
    def __init__(self, save_path):
        self.session = tf.Session()
        self.save_path = save_path

        self.define_inputs()
        self.define_fc1()
        self.define_fc2()
        self.define_softmax()
        self.define_optimization()

    def weight_variable(self, shape):
        w_init = tf.truncated_normal(shape, stddev=0.01)
        return tf.get_variable('w', initializer=w_init)

    def bias_variable(self, shape):
        b_init = tf.constant(0.0, shape=shape)
        return tf.get_variable('b', initializer=b_init)

    def define_inputs(self):
        with tf.variable_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, 784])
            self.y = tf.placeholder(tf.float32, [None, 10])
            self.prob_i = tf.placeholder(tf.float32)
            x_drop = tf.nn.dropout(self.x, keep_prob=self.prob_i)

    def define_fc1(self):
        with tf.variable_scope('fc1'):
            W_fc1 = self.weight_variable([784, 1200])
            b_fc1 = self.bias_variable([1200])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.x, W_fc1) + b_fc1)
            self.prob_fc = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(self.h_fc1, self.prob_fc)

    def define_fc2(self):
        with tf.variable_scope('fc2'):
            W_fc2 = self.weight_variable([1200, 1200])
            b_fc2 = self.bias_variable([1200])
            self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, W_fc2) + b_fc2)
            h_fc2_drop = tf.nn.dropout(self.h_fc2, self.prob_fc)

    def define_softmax(self):
        with tf.variable_scope('softmax'):
            W_fc3 = self.weight_variable([1200, 10])
            b_fc3 = self.bias_variable([10])
            self.preds = tf.nn.softmax(tf.matmul(self.h_fc2, W_fc3) + b_fc3)

    def define_optimization(self):
        with tf.variable_scope('optimization'):
            # define the loss function
            self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.preds), reduction_indices=[1]))
            # define training step and accuracy
            self.train_step = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(self.cross_entropy)
            self.correct = tf.equal(tf.argmax(self.preds, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

    def training_accuracy(self):
        return self.session.run(self.accuracy, feed_dict={self.x: self.x_train, self.y: self.y_train,
            self.prob_i: 1.0, self.prob_fc: 1.0})

    def validation_accuracy(self):
        return self.session.run(self.accuracy, feed_dict={self.x: mnist.validation.images,
            self.y: mnist.validation.labels, self.prob_i: 1.0, self.prob_fc: 1.0})

    def test_accuracy(self):
        feed_dict={self.x: mnist.test.images,
                        self.y: mnist.test.labels, self.prob_i: 1.0, self.prob_fc: 1.0}
        test_accuracy = self.session.run(self.accuracy, feed_dict={self.x: mnist.test.images,
                        self.y: mnist.test.labels, self.prob_i: 1.0, self.prob_fc: 1.0})
        return test_accuracy

    def train(self, num_iterations):
        with self.session as sess:
            self.saver = tf.train.Saver(tf.global_variables())
            batch_size = 100
            print('Starting training...')
            start_time = time.time()
            best_accuracy = 0
            init = tf.global_variables_initializer()
            self.session.run(init)

            for i in range(num_iterations):
                self.x_train, self.y_train = mnist.train.next_batch(batch_size)
                if (i + 1) % 1000 == 0:
                    train_accuracy = self.training_accuracy()
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                #run training step (note dropout hyperparameters)
                sess.run(self.train_step, feed_dict={self.x: self.x_train,
                    self.y: self.y_train, self.prob_i: 0.8, self.prob_fc: 0.5})

                # validate
                val_accuracy = self.validation_accuracy()
                print("val_accuracy %g" % val_accuracy)

                if val_accuracy > best_accuracy:
                    self.saver.save(sess, self.save_path)
                    best_accuracy = val_accuracy
                    print("Validation accuracy improved: %g. Saving the network." % val_accuracy)
                else:
                    self.saver.restore(sess, self.save_path)
                    print("Validation accuracy was: %g. Previous accuracy: %g. " % (val_accuracy, best_accuracy) + "Using old parameters for further optimizations.")

            print("Training took %.4f seconds." % (time.time() - start_time))
            self.test_accuracy()
            print("Best test accuracy: %g" % best_accuracy)

    """
    def load_trained_model(self):
        tf.reset_default_graph()
        self.session = tf.InteractiveSession()
        self.saver = tf.train.import_meta_graph('/models/fcn/mnist_fc.meta')
        self.saver.restore(self.session, tf.train.latest_checkpoint('../models/fcn'))
        self.x = tf.get_default_graph().get_tensor_by_name('inputs/Placeholder:0')
        self.preds = tf.get_default_graph().get_tensor_by_name('softmax/Softmax:0')
        p1 = tf.get_default_graph().get_tensor_by_name('inputs/Placeholder_2:0')
        p2 = tf.get_default_graph().get_tensor_by_name('fc1/Placeholder:0')
    """

    def plot_predictions(self, *args):
        self.session = tf.InteractiveSession()
        self.saver.restore(self.session, self.save_path)
        digit_indices = np.array(list(args))
        num_indices = len(digit_indices)
        fig, axes =  plt.subplots(ncols=num_indices, figsize=(num_indices, 1))
        p1 = tf.get_default_graph().get_tensor_by_name('inputs/Placeholder_2:0')
        p2 = tf.get_default_graph().get_tensor_by_name('fc1/Placeholder:0')
        for i, idx in enumerate(digit_indices):
            ax = axes.flatten()[i]
            vec = mnist.test.images[idx]
            img = vec.reshape(28, 28)
            args = {self.x: np.expand_dims(vec, 0), p1: 1.0, p2: 1.0}
            pred = self.session.run(self.preds, feed_dict=args)
            ax.set_title('Pred: {}'.format(np.argmax(pred)))
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        plt.show()


if __name__ == "__main__":
    #print_tensors_in_checkpoint_file('models/fcn/mnist_fc', '', '', True)
    #plot_digits(5, 33, 1445, 3434, 9888)
    net = FCN('models/fcn/mnist_fc')
    net.train(100)
    # max 10,000
    net.plot_predictions(5, 33, 1445, 3434, 9888)
