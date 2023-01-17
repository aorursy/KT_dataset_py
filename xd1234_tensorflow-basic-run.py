# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

images = data.iloc[:, 1:].values

#print (images[0,:])
labels = data.iloc[:, 0].values

#print (len(label))
from sklearn.preprocessing import LabelBinarizer



labels = LabelBinarizer().fit_transform(labels)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.25, random_state = 0)
import tensorflow as tf



x = tf.placeholder(tf.float32, [None, 784])

w = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)
y1 = tf.placeholder(tf.float32, [None, 10])

entropy = -tf.reduce_sum(y1 * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(entropy)
epochs_completed = 0

index_in_epoch = 0

num_examples = train_images.shape[0]



# serve data by batches

def next_batch(batch_size):

    

    global train_images

    global train_labels

    global index_in_epoch

    global epochs_completed

    

    start = index_in_epoch

    index_in_epoch += batch_size

    

    # when all trainig data have been already used, it is reorder randomly    

    if index_in_epoch > num_examples:

        # finished epoch

        epochs_completed += 1

        # shuffle the data

        perm = np.arange(num_examples)

        np.random.shuffle(perm)

        train_images = train_images[perm]

        train_labels = train_labels[perm]

        # start next epoch

        start = 0

        index_in_epoch = batch_size

        assert batch_size <= num_examples

    end = index_in_epoch

    return train_images[start:end], train_labels[start:end]
init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

for i in range(1000):

    batch_xs, batch_ys = next_batch(100)

    sess.run(train_step, feed_dict={x: batch_xs, y1: batch_ys})
data1 = pd.read_csv('../input/test.csv')

print (data1)


#test_images = data1.iloc[:, 1:].values

#test_labels = data1.iloc[:, 0].values

#test_labels = LabelBinarizer().fit_transform(test_labels)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y1,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print (sess.run(accuracy, feed_dict={x: test_images, y1: test_labels}))