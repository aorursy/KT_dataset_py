import tensorflow as tf
import pandas as pd
import numpy as np
print("Hello Kaggle!")
print("This is a test.")
#Tensorflow
hello = tf.constant('Hello, TensorFlow!')
a = tf.constant(10)
b = tf.constant(32)
add = a + b

sess = tf.Session()
print(sess.run(hello))
print(sess.run(add))
