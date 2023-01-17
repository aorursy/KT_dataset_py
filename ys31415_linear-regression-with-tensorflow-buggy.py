import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf
x_data = np.random.uniform(high=10,low=0,size=100)

y_data = 3.5 * x_data -4 + np.random.normal(loc=0, scale=2,size=100)



plt.plot(x_data, y_data, linestyle="", marker=".")
X = tf.placeholder(dtype=tf.float32, shape=100)

Y = tf.placeholder(dtype=tf.float32, shape=100)

m = tf.Variable(1.0)

c = tf.Variable(1.0)

Ypred = m*X + c

loss = tf.reduce_mean(tf.square(Ypred - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1)

train = optimizer.minimize(loss)
session = tf.Session()

session.run(tf.global_variables_initializer())
convergenceTolerance = 0.0001

previous_m = np.inf

previous_c = np.inf



steps = {}

steps['m'] = []

steps['c'] = []



losses=[]



for k in range(100):

    _m = session.run(m)

    _c = session.run(c)

    _l = session.run(loss, feed_dict={X: x_data, Y:y_data})

    session.run(train, feed_dict={X: x_data, Y:y_data})

    steps['m'].append(_m)

    steps['c'].append(_c)

    losses.append(_l)

    if (np.abs(previous_m - _m) or np.abs(previous_c - _c) ) <= convergenceTolerance :        

        print("Finished by Convergence Criterion")

        print(k)

        print(_l)

        break

    previous_m = _m, 

    previous_c = _c,



    

session.close()
plt.plot(losses)