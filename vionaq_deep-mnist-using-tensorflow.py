import tensorflow as tf

import pandas as pd

import numpy as np



mnist = pd.read_csv('../input/train.csv')

mnist_test = pd.read_csv('../input/test.csv')



x = tf.placeholder(tf.float32, [None, 784])

y_ = tf.placeholder(tf.float32, [None, 10])
def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev = 0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)
def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool(x):

    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#initailizing the weight and bias variables

W_conv1 = weight_variable([5, 5, 1, 32])

b_conv1 = bias_variable([32])

#reshaping the input variable to a 4d tensor

x_image = tf.reshape(x, [-1, 28, 28, 1])

#convolution and max pooling

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool(h_conv1)
#initializing the weight and bias variables

W_conv2 = weight_variable([5, 5, 32, 64])

b_conv2 = bias_variable([64])



#convolution and max pooling

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool(h_conv2)
W_fc1 = weight_variable([7*7*64, 1024])

b_fc1 = bias_variable([1024])



h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#prob that neuron's output should be kept during dropout

keep_prob = tf.placeholder(tf.float32)

h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])

b_fc2 = bias_variable([10])



y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2
#Function to randomly select batches of training data

def next_batch(num):

    l = np.random.randint(0, 42000, num)

    return (mnist.iloc[l, 1:], np.eye(10)[mnist.iloc[l, 0]])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

predictions = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y_conv, 1))

accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))

result = tf.arg_max(y_conv, 1)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

with sess.as_default():

    for i in range(1000):

        batch = next_batch(50)

        if i%100 == 0:

            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

            print ("step %d has accuracy %g"%(i, train_accuracy))

        train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    low = 0

    arr = []

    while (low <= (28000-50)):

        arr.extend(result.eval(feed_dict={x:mnist_test.iloc[low:low+50, :], keep_prob:1.0}).tolist())

        #print("Step: \n" + str(low))

        low += 50

    

    y_test_labels = pd.DataFrame({"ImageId": list(range(1,len(arr)+1)),'Label': arr})

    #y_test_labels.index += 1

    #y_test_labels.columns.names = ['ImageId']

    print (y_test_labels)

#from subprocess import check_output



y_test_labels.to_csv('output.csv', index=False, header=True, sep = ',')

#print(check_output(["ls", "./"]).decode("utf8"))