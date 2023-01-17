# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train_X = train.ix[:, 1:]

train_y = train.ix[:, 0]

train_y = train_y.astype(float)

train_y = train_y.tolist()



test_X = test
train_X -= np.mean(train_X, axis=0)

test_X -= np.mean(test_X, axis=0)



batch_size = 16

image_size = 28

num_label = 10

filter_size = 5

depth = 16

num_hidden = 64

num_channel = 1



train_y = tf.one_hot(train_y, 10, on_value=1.0, off_value=0.0, dtype=tf.float32)



with tf.Session() as sess:

    # transfer tensor back to numpy array

    train_y = sess.run(train_y)



def accuracy(pred_y, true_y):

    return (np.sum(np.argmax(pred_y, 1) == np.argmax(true_y, 1)) / float(pred_y.shape[0])) * 100.