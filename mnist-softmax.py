# by importing tensorflow an empty default graph is created
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = 'datasets/mnist'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

# one_hot means that the labels will be presented in a way that only one bit will be on for a specific digit.
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

# x: will contain MINIBATCH_SIZE rows of 28x28=784 float32 pixel gray values representing written digits
x = tf.placeholder(tf.float32, [None, 784])

# W: weight matrix of 784 rows and 10 columns will be trained by GradientDecentOptimizer
W = tf.Variable(tf.zeros([784, 10]))

# y_true: will contain MINIBATCH_SIZE rows times 10 columns
# defining the true value of the written digit with a 1.0 in the digit's column (0.0 otherwise)
y_true = tf.placeholder(tf.float32, [None, 10])

# y_pred: will contain MINIBATCH_SIZE rows times 10 columns
# each column contains the estimated probability that the column's index is the written digit
y_pred = tf.matmul(x, W)

# compute the cross entropy of the predicted results and the true results in the MINIBATCH
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

# optimize the weights with a 0.5 step towards the gradient of the cross_entropy
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# argmax returns the index with the largest value of the 10 columns across the batch labels of y_pred and y_true.
# The prediction is correct if for a label y_pred has the highest value in the column that is marked with a 1 in y_true.
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
# since tf.equal returns a bool type it must be casted to a float32 in order to compute the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# Create the Tensorflow Session and use it as a context manager in a with clause in order to be closed appropriately.
with tf.Session() as sess:
    # Train

    # initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # start the training of the weights by stepping down the gradients of the cross_entropy of the batches
    for step in range(NUM_STEPS):
        # load the data of the next batch into batch_xs and batch_ys
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        # feed the placeholders x and y_true with batch_xs and batch_ys, respectively.
        # Compute the cross_entropy and the accuracy of the step
        (_, batch_cross_entropy, batch_accuracy) = sess.run([gd_step, cross_entropy, accuracy],
                                                            feed_dict={x: batch_xs, y_true: batch_ys})
        print("[step {}] cross entropy of training batch: {:.4f}, accuracy: {:.2f}%"
              .format(step, batch_cross_entropy, batch_accuracy * 100))

    # Test

    # load all 10000 existing images and labels of the test sets and compute the accuracy
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("accuracy of test set: {:.2f}%".format(ans * 100))
