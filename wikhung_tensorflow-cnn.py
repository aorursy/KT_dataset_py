# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
path = '../input/'

train_df = pd.read_csv(path + 'train.csv')

test_df = pd.read_csv(path + 'test.csv')
train_df.head(10)
train_x, train_y = train_df.ix[:,1:train_df.shape[1]].as_matrix(), train_df.ix[:,0].as_matrix()

test_x = test_df.as_matrix()
train_x, test_x = train_x / 255., test_x / 255.
train_x = train_x.reshape([-1, 28, 28, 1])

test_x = test_x.reshape([-1, 28, 28, 1])
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

train_y = enc.fit_transform(train_y.reshape(-1,1)).toarray()
from sklearn.model_selection import ShuffleSplit



seed = 2017

splitter = ShuffleSplit(test_size = .1, random_state = seed)
train_idx, val_idx = next(splitter.split(train_x))
val_x, val_y = train_x[val_idx], train_y[val_idx]

train_x, train_y = train_x[train_idx], train_y[train_idx]
import matplotlib.pyplot as plt

plt.figure()

plt.imshow(train_x[0].reshape(28, 28), cmap = 'gray')

plt.figure()

plt.imshow(train_x[1].reshape(28, 28), cmap = 'gray')
import tensorflow as tf



input_ = tf.placeholder(tf.float32, (None, 28, 28,1), name = 'features')

label = tf.placeholder(tf.int32, (None, 10), name = 'labels')
def weight_variable(shape):

    init = tf.truncated_normal(shape, stddev = 0.1)

    return tf.Variable(init)



def bias_variable(shape):

    init = tf.constant(0.1, shape = shape)

    return tf.Variable(init)
def conv2d(x, conv_filter, stride, padding):

    return tf.nn.conv2d(x, conv_filter, strides = [1, stride, stride, 1], padding = padding)



def max_pool(x, k, stride, padding):

    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, stride, stride, 1],

                          padding = padding)
def CNN(x):

    filter1_depth = 15

    conv_filter1 = weight_variable([2, 2, 1, filter1_depth])

    conv_bias1 = bias_variable([filter1_depth])

    

    conv_layer1 = conv2d(x, conv_filter1, 2, 'SAME')

    conv_layer1 = tf.nn.bias_add(conv_layer1, conv_bias1)

    conv_layer1 = tf.nn.relu(conv_layer1)

    

    max_pool1 = max_pool(conv_layer1, 2, 2, 'SAME')

    

    filter2_depth = 30

    conv_filter2 = weight_variable([4, 4, filter1_depth, filter2_depth])

    conv_bias2 = bias_variable([filter2_depth])

    

    conv_layer2 = conv2d(max_pool1, conv_filter2, 2, 'SAME')

    conv_layer2 = tf.nn.bias_add(conv_layer2, conv_bias2)

    conv_layer2 = tf.nn.relu(conv_layer2)

    

    max_pool2 = max_pool(conv_layer2, 2, 2, 'SAME')

    

    # Shortcut

    #flatten_layer = tf.contrib.layers.flatten(max_pool2)

    #fc = tf.contrib.layers.full_connected(flatten_layer, 256)

    

    # Long way of doing the fully_connected_layer

    _, height, width, depth = max_pool2.shape

    tensor_size = int(height * width * depth)

    flatten_layer = tf.reshape(max_pool2, [-1, tensor_size])

    

    fc1_hidden_unit = 256

    W_fc1 = weight_variable([tensor_size, fc1_hidden_unit])

    b_fc1 = bias_variable([fc1_hidden_unit])

    

    fc1 = tf.nn.relu(tf.add(tf.matmul(flatten_layer, W_fc1), b_fc1))

                    

    n_class = 10

    W_fc2 = weight_variable([fc1_hidden_unit, n_class])

    b_fc2 = bias_variable([n_class])

                     

    logits = tf.add(tf.matmul(fc1, W_fc2), b_fc2)

    

    return logits
#conv_filter = tf.Variable(tf.truncated_normal((2, 2, 1, 15)))

#conv_layer = tf.nn.conv2d(input_, conv_filter, [1,2,2,1], padding = 'SAME')
#flatten_layer = tf.contrib.layers.flatten(conv_layer)

#fc = tf.contrib.layers.fully_connected(flatten_layer, 256)

#output = tf.contrib.layers.fully_connected(fc, 10, activation_fn = None)
logits = CNN(input_)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logits)

loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer().minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)), tf.float32))
def create_batches(x, y, n_batch, batch_size):

    if (n_batch + 1) * batch_size > x.shape[0]:

        end_idx = n_batch * batch_size

    else:

        end_idx = (n_batch + 1) * batch_size

    

    return(x[:end_idx].reshape(-1, batch_size, 28, 28, 1), y[:end_idx].reshape(-1, batch_size, 10))
epoch = 10

batch_size = 32

n_batch = int(np.floor(train_x.shape[0] / batch_size))



tf.get_default_graph()

sess = tf.InteractiveSession()



sess.run(tf.global_variables_initializer())

for e in range(epoch):

    print('Epoch {0}'.format(e))

    batch_x, batch_y = create_batches(train_x, train_y, n_batch, batch_size)

    for batch in range(n_batch):

        sess.run([optimizer, loss], feed_dict = {input_: batch_x[batch],

                                                 label: batch_y[batch]} )



        if batch % 100 == 0:

            val_feed_dict = {input_: val_x,

                             label: val_y}

            val_loss = sess.run(loss, feed_dict = val_feed_dict)

            val_acc = accuracy.eval(feed_dict = val_feed_dict)

            print("Batch #{}: Valditaion Loss: {:.3}, Validation Accuracy: {:.3}"\

                  .format(batch, val_loss, val_acc))