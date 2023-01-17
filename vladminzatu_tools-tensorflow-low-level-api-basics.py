# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
a = tf.constant([[1.0, 2.0]], dtype=tf.float32)
b = tf.constant([[2.0],[3.0]])
result = tf.matmul(a, b)
print(a,b,result)
arr = np.array([[1,2,3],[4,5,6]])
c = tf.convert_to_tensor(arr)
print(c)
with tf.Session() as sess:
    print(sess.run([result, c]))
c = tf.placeholder(tf.float32, shape=[3])
with tf.Session() as sess:
    print(sess.run(c, {c: [1.0, 2, 3]}))
x = tf.placeholder(tf.int64, shape=[None, 2])
print("The shape is: ", x.shape)
with tf.Session() as sess:
    print(sess.run(x, {x: [[1, 2]]}))
    print(sess.run(x, {x: [[1, 2],[3,4]]}))
my_var = tf.get_variable("my_var", shape = [1, 2, 3])
with tf.Session() as sess:
    sess.run(my_var.initializer)
    sess.run(tf.global_variables_initializer())
    print(sess.run(my_var))
