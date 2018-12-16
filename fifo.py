import threading

import tensorflow as tf

# create a FIFOQueue that produces a range of numbers several times and register an associated QueueRunner
number_queue = tf.train.input_producer([i for i in range(20)], num_epochs=5, name='number_queue')

# manually build a FIFOQueue, and a QueueRunner
queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], name='consumer_queue')
enqueue_op = queue.enqueue(tf.as_string(number_queue.dequeue()))
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
dequeue_op = queue.dequeue()
is_closed_op = queue.is_closed()

# Thread coordinator
coord = tf.train.Coordinator()


def dequeue_and_print(sess, queue, coord):
    while not coord.should_stop():
        queue_is_closed = sess.run(is_closed_op)
        if (queue_is_closed):
            coord.request_stop()
            continue
        print("Queue:")
        size = sess.run(queue.size())
        for line in range(size):
            dequeue_val = sess.run(dequeue_op)
            print("{}: {}".format(line, dequeue_val.decode("utf-8")))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tf.train.add_queue_runner(qr)

    # create and start the QueueRunners
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # start dequeue_and_print thread
    print_thread = threading.Thread(target=dequeue_and_print, args=[sess, queue, coord])
    coord.register_thread(print_thread)
    print_thread.start()

    # wait until all threads are done
    coord.join(threads)
