import os

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

DATA_PATH = 'datasets/mnist'
NUM_EPOCHS = 5000
BATCH_SIZE = 128


def download_mnist():
    data_sets = mnist.read_data_sets(DATA_PATH, dtype=tf.uint8, reshape=False, validation_size=1000)
    data_splits = ['train', 'test', 'validation']
    for d, data_split in enumerate(data_splits):

        filename = os.path.join(DATA_PATH, data_split + '.tfrecords')
        if os.path.isfile(filename):
            # skip processing this split if tfrecord already exists
            continue

        print('Saving ' + data_split)
        data_set = data_sets[d]

        with tf.python_io.TFRecordWriter(filename) as writer:
            for index in range(data_set.images.shape[0]):
                image = data_set.images[index].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[1]])),
                    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[2]])),
                    'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[3]])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(data_set.labels[index])])),
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }))
                writer.write(example.SerializeToString())


download_mnist()

filename = os.path.join(DATA_PATH, 'train.tfrecords')
# operation to put the filename 10 times into a queue
filename_queue = tf.train.string_input_producer([filename], num_epochs=10)

# TFRecordReader consumes the filenames from the filename_queue and read its contents into serialized_example
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

# features is a dictionary of operations parsed from the serialized_example
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
)

# operation to decode the raw image into an operation of unknown shape
image = tf.decode_raw(features['image_raw'], tf.uint8)
# give image the shape of the flat 784 uint8 values
image.set_shape([784])
# convert image into 785 float32 values between -0.5 and 0.5
image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
# cast labels from int64 to int32
label = tf.cast(features['label'], tf.int32)

# images_batch and labels_batch each contain up to 128 images consisting of 784 float32 numbers
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=BATCH_SIZE,
    capacity=2000,
    min_after_dequeue=1000
)

W = tf.get_variable("W", [28 * 28, 10])
y_pred = tf.matmul(images_batch, W)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

# argmax returns the index with the largest value of the 10 columns across the batch labels of y_pred and y_true.
# The prediction is correct if for a label y_pred has the highest value in the column that is marked with a 1 in y_true.
correct_mask = tf.equal(tf.argmax(y_pred, 1, output_type=tf.int32), labels_batch)
# since tf.equal returns a bool type it must be casted to a float32 in order to compute the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# Create the Tensorflow Session and use it as a context manager in a with clause in order to be closed appropriately.
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    init = tf.local_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = 0
    try:
        while not coord.should_stop():
            step += 1
            sess.run([train_op])
            if step % 500 == 0:
                (accuracy_val, cross_entropy_val) = sess.run([accuracy, loss_mean])
                print("[step {}] cross entropy of training batch: {:.4f}, accuracy: {:.2f}%"
                      .format(step, cross_entropy_val, accuracy_val * 100))
    except tf.errors.OutOfRangeError:
        print('Training finished for {} epochs, {} steps.'.format(NUM_EPOCHS, step))
    finally:
        coord.request_stop()

    coord.join(threads)
