# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # TensorFlow framework



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

% matplotlib inline

# Any results you write to the current directory are saved as output.
testdata = pd.read_csv('../input/test.csv')

traindata = pd.read_csv('../input/train.csv')
def preprocessing(X):

    X_processed = X/255

    X_processed = X_processed

    return X_processed
Y_train = traindata['label']

X_train = traindata.loc[:,'pixel0':]
X_train = np.array(preprocessing(X_train))

Y_train = np.eye(10)[Y_train]
X_test = preprocessing(testdata)
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')



Y = tf.placeholder(tf.float32, shape=[None, 10], name='Y')
tf.set_random_seed(1)

W1 = tf.Variable(np.random.rand(X.shape[1], 12) * 0.01, dtype=tf.float32)

b1 = tf.Variable(np.zeros([12]), dtype=tf.float32)

z1 = tf.add(tf.matmul(X, W1), b1)

A1 = tf.nn.relu(z1)

W2 = tf.Variable(np.random.rand(12, 5) * 0.01, dtype=tf.float32)

b2 = tf.Variable(np.zeros((1, 5)), dtype=tf.float32)

z2 = tf.add(tf.matmul(A1, W2), b2)

A2 = tf.nn.relu(z2)

W3 = tf.Variable(np.random.rand(5, 10) * 0.01, dtype=tf.float32)

b3 = tf.Variable(np.zeros((1, 10)), dtype=tf.float32)

z3 = tf.add(tf.matmul(A2, W3), b3)
z3.shape
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=z3))
def mini_batch_generate(X, Y, N):

    X_batch = []

    Y_batch = []

    index = np.arange(X.shape[0])

    np.random.shuffle(index)

    X_shuffled = X[index]

    Y_shuffled = Y[index]

    n = X.shape[0]//N

    for i in range(n):

        X_chunk = X_shuffled[i*N: (i+1)*N]

        Y_chunk = Y_shuffled[i*N: (i+1)*N]

        X_batch.append(X_chunk)

        Y_batch.append(Y_chunk)

    if X.shape[0] > (n*N):

        X_chunk = X_shuffled[(i+1)*N:]

        Y_chunk = Y_shuffled[(i+1)*N:]

        X_batch.append(X_chunk)

        Y_batch.append(Y_chunk)

        n = n + 1

    return X_batch, Y_batch, n
cost_record = []

train_step = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(800):

        np.random.seed(i+1)

        mini_batch_X, mini_batch_Y, n = mini_batch_generate(X_train, Y_train, 128)

        epoch_cost = 0

        for j in range(n):

            _, c = sess.run([train_step, cost], feed_dict={X: mini_batch_X[j], Y: mini_batch_Y[j]})

            epoch_cost += c/n

        if i%100 == 0:

            print ('Cost', epoch_cost)

        if i%5 == 0:

            cost_record.append(epoch_cost)

    correct_prediction = tf.equal(tf.argmax(z3, 1), tf.argmax(Y, 1))



    # Calculate accuracy on the test set

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



    print ("Train Accuracy:", accuracy.eval(feed_dict={X: X_train, Y: Y_train}))

    plt.plot(cost_record)

    y_test = tf.nn.softmax(z3)

    result = sess.run(y_test, feed_dict={X: X_test})

    
r = [result[i].argsort()[0] for i in range(result.shape[0])]

pd.DataFrame(r, columns=['Num'])

submission = pd.DataFrame({"ImageID": [i for i in range(result.shape[0])], "Label": r})

submission.to_csv("MNIST_Adam.csv", index=False)



import os

os.getcwd()