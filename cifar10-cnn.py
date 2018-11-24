import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_PATH = 'datasets/cifar-10-batches-py'
STEPS = 5000
BATCH_SIZE = 100


def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d[b"data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.labels = one_hot(np.hstack([d[b"labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i + batch_size], self.labels[self._i:self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y


class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)]).load()
        self.test = CifarLoader(["test_batch"]).load()


def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) for i in range(size)])
    plt.imshow(im)
    plt.show()


cifar = CifarDataManager()
print("Number of training images: {}".format(len(cifar.train.images)))
print("Number of training labels: {}".format(len(cifar.train.labels)))
print("Number of test images: {}".format(len(cifar.test.images)))
print("Number of test labels: {}".format(len(cifar.test.labels)))
images = cifar.train.images
display_cifar(images, 10)


#############################
# Construction of the CNN

# initialization of weights of a layer
def weight_variable(shape):
    # truncated normal distribution with standard deviation of 0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# initialization of bias variable of a layer to 0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b


# placeholder for the images of the digits
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

# one hot representation of the true digit
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# first convolution using a 5x5 kernel transforming 3 feature maps (RGB) into to 32
conv1 = conv_layer(x, shape=[5, 5, 3, 32])
# followed by a 2x2 max pooling layer resulting in a 16x16x32 tensor
conv1_pool = max_pool_2x2(conv1)

# second convolution using a 5x5 kernel transforming 32 feature maps into to 64
conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
# followed by a 2x2 max pooling layer resulting in a 8x8x64 tensor
conv2_pool = max_pool_2x2(conv2)

# reshape output of second convolution to a linear tensor
conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])
# and fully connect to a fully connected layer with 1024 outputs
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

# feed the 1x1024 layer into a dropout layer with the given probability to keep values
keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

# the final fully connected layer reduces the 1024 values to 10 as the predicted one hot representation of the digit
y_conv = full_layer(full1_drop, 10)

# end of CNN construction
#############################


# compute the cross entropy of the predicted results and the true results in the MINIBATCH
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# minimize the cross entropy with an adam optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# argmax returns the index with the largest value of the 10 columns across the batch labels of y_conv and y_.
# The prediction is correct if for a label y_conv has the highest value in the column that is marked with a 1 in y_.
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# since tf.equal returns a bool type it must be casted to a float32 in order to compute the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def test(sess):
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    test_accuracy = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(10)])
    print("accuracy of test set: {:.2f}%".format(test_accuracy * 100))


# Create the Tensorflow Session and use it as a context manager in a with clause in order to be closed appropriately.
with tf.Session() as sess:
    # initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # start the training of the weights by stepping through the batches
    for i in range(STEPS):
        # load the data of the next batch into batch_xs and batch_ys
        batch_xs, batch_ys = cifar.train.next_batch(BATCH_SIZE)

        if i % 100 == 0:
            batch_cross_entropy, batch_accuracy = sess.run([cross_entropy, accuracy],
                                                           feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print("[step {}] cross entropy of training batch: {:.4f}, accuracy: {:.2f}%"
                  .format(i, batch_cross_entropy, batch_accuracy * 100))

        # feed the placeholders x and y_ with batch_xs and batch_ys, respectively.
        # Compute the cross_entropy and the accuracy of the step
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    test(sess)
