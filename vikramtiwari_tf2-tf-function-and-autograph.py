import sys
print("Python version:", sys.version)

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from __future__ import absolute_import, division, print_function
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
@tf.function
def simple_nn_layer(x, y):
    return tf.nn.relu(tf.matmul(x, y))

x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

simple_nn_layer(x, y)
simple_nn_layer
def linear_layer(x):
    return 2 * x + 1

@tf.function
def deep_net(x):
    return tf.nn.relu(linear_layer(x))

deep_net(tf.constant((1, 2, 3)))
@tf.function
def square_if_positive(x):
    if x > 0:
        x = x * x
    else:
        x = 0
    return x

print('square_if_positive(2) = {}'.format(square_if_positive(tf.constant(2))))
print('square_if_positive(-2) = {}'.format(square_if_positive(tf.constant(-2))))
@tf.function
def sum_even(items):
    s = 0
    for c in items:
        if c % 2 > 0:
            continue
        s += c
    return s

sum_even(tf.constant([10, 12, 15, 20]))
print(tf.autograph.to_code(sum_even.python_function, experimental_optional_features=None))
@tf.function
def fizzbuzz(n):
    msg = tf.constant('')
    for i in range(n):
        if i % 3 == 0:
            msg += 'Fizz'
        elif i % 5 == 0:
            msg += 'Buzz'
        else:
            msg += tf.as_string(i)
        msg += '\n'
    return msg

print(fizzbuzz(tf.constant(15)).numpy().decode())
@tf.function
def count(n):
    for i in tf.range(n):
        print(i)
        
count(tf.constant(5))
@tf.function
def range_example(n):
    return range(n)

print(range_example(tf.constant(3)))
@tf.function
def len_example(n):
    return len(n)

print(len_example(tf.zeros((20, 10))))
class CustomModel(tf.keras.models.Model):
    
    @tf.function
    def call(self, input_data):
        if tf.reduce_mean(input_data) > 0:
            return input_data
        else:
            return input_data // 2
        
model = CustomModel()
model(tf.constant([-2, -4]))
v = tf.Variable(5)

@tf.function
def find_next_odd():
    v.assign(v + 1)
    if v % 2 == 0:
        v.assign(v + 1)

find_next_odd()
v
def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    
    return x, y

def mnist_dataset():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds

train_dataset = mnist_dataset()
model = tf.keras.Sequential((
    tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10)))

model.build()
optimizer = tf.keras.optimizers.Adam()
def compute_loss(logits, labels):
    return tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(predictions == labels, tf.float32))

def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        tape.watch(model.variables)
        logits =  model(x)
        loss = compute_loss(logits, y)
        
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))
    
    accuracy = compute_accuracy(logits, y)
    return loss, accuracy


@tf.function
def train(model, optimizer):
    train_ds = mnist_dataset()
    step = 0
    for x, y in train_ds:
        step += 1
        loss, accuracy = train_one_step(model, optimizer, x, y)
        if step % 10 == 0:
            print('Step', step, ': loss', loss, ':, accuracy', accuracy)
    return step

_ =  train(model, optimizer)
def square_if_positive(x):
    return [i ** 2 if i > 0 else i for i in x]

square_if_positive(range(-5, 5))
@tf.function
def square_if_positive_naive(x):
    result = tf.TensorArray(tf.int32, size=len(x))
    for i in range(len(x)):
        if x[i] > 0:
            result = result.write(i, x[i] ** 2)
        else:
            result = result.write(i, x[i])
    return result.stack()

square_if_positive_naive(tf.range(-5, 5))
def square_if_positive_vectorized(x):
    return tf.where(x > 0, x ** 22, x)

square_if_positive_vectorized(tf.range(-5, 5))