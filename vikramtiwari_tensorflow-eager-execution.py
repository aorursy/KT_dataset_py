from __future__ import absolute_import, division, print_function
import tensorflow as tf
# eager mode is available in core since 1.7.0
tf.VERSION
# enabled eager mode execution
tf.enable_eager_execution()
# check if it's set properly
tf.executing_eagerly()
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))
# Evaluation, printing and checking tensor values doesn't break the flow for computing gradients
# works nicely with NumPy
a = tf.constant([[1, 2],
                 [3, 4]])
print('a:', a)

b = tf.add(a, 1)
print('\nb:', b)

# operator overloading is supported
print('\na * b', a * b)

import numpy as np

c = np.multiply(a, b)
print('\nc:', c)

# obtain numpy values from a tensor
print('\na.numpy():', a.numpy())
# Performance
import time

def measure(x, steps):
    # TensorFlow initializes a GPU the first time it's used, exclude from timing.
    tf.matmul(x, x)
    start = time.time()
    for i in range(steps):
        x = tf.matmul(x, x)
        _ = x.numpy()  # Make sure to execute op and not just enqueue it
    end = time.time()
    return end - start

shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

# Run on CPU:
with tf.device("/cpu:0"):
    print("CPU: {} secs".format(measure(tf.random_normal(shape), steps)))

# Run on GPU, if available:
if tfe.num_gpus() > 0:
    with tf.device("/gpu:0"):
        print("GPU: {} secs".format(measure(tf.random_normal(shape), steps)))
else:
    print("GPU: not found")
# If you wanna use eager in a graph environment, use `tfe`
# NOTE: Only works if `tf.enable_eager_execution()` hasn't been called yet
import tensorflow.contrib.eager as tfe

def my_py_func(x):
    x = tf.matmul(x, x) # tf ops
    print(x)  # but it's eager!
    return x

with tf.Session() as sess:
    x = tfe.placeholder(dtype=tf.float32)
    # Call eager function in graph!
    pf = tfe.py_func(my_py_func, [x], tf.float32)
    sess.run(pf, feed_dict={x: [[2.0]]})  # [[4.0]]