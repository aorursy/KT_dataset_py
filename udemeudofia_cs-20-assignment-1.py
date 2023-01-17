# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
sess = tf.InteractiveSession()
x=3
y=4
a=tf.multiply(x,x)
b=tf.multiply(a,y)
c=tf.add(y,2)
i=tf.add(b,c)

print(sess.eval(i))
sess.close()
p = tf.random_uniform([])
q = tf.random_uniform([])
out = tf.cond(tf.greater(p,q), lambda:p+q, lambda:p-q)
print(out.eval())

f = tf.random_uniform([], -1,1, dtype=tf.float32)
g = tf.random_uniform([], -1,1, dtype=tf.float32)
out = tf.case({tf.less(f,g): lambda: tf.add(f,g), tf.greater(f,g): lambda: tf.subtract(f,g)}, default=lambda: tf.constant(0.0))
#print(sess.run(out))
a = tf.constant([[0, -2, -1], [0, 1, 2]])
b = tf.zeros_like(a)
op = tf.equal(a,b)
print(sess.run(op))
