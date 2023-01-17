!pip install tensorflow==2.0.0-alpha0



import tensorflow as tf

from tensorflow.python.ops import control_flow_util

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
print(tf.__version__)
print(tf.executing_eagerly())
x = [[2.]]

tf.matmul(x, x)
print(tf.constant([[2.0],[1.0]]))
x = tf.constant(1.0)

x *= 10

print(x)

print(x.numpy())
print(tf.add(x, [[1.0],[2.0]]))
v = tf.Variable(tf.random.normal([2, 1]))
v.assign([[1.0], [2.0]])
@tf.function

def add_tensors(x, y):

    return tf.add(x, y)
# not even gonna bother with tf.constant. Much convenient, much eager

x = [1.0, 2.0]

y = [2.0, 3.0]

add_tensors(x, y)
@tf.function

def simple_abs(x):

    if x < 0:

        -x

    else:

        x

print(tf.autograph.to_code(simple_abs.python_function, experimental_optional_features=None))