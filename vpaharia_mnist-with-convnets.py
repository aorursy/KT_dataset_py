import numpy as np

import tensorflow as tf

import pandas as pd
images = pd.read_csv('../input/train.csv')
X = images[images.columns[1:]].as_matrix()



Y = np.zeros([len(images),10])

for i in range(len(images)):

   Y[i][images.iloc[i]['label']] = 1
#Hyper Parameters

samples = images.count()[0]



epochs = 20 #increase this to get better accuracy

starter_learning_rate = 0.01 #starting learning rate, decays exponentially

batch_size = 100
#placeholders

x = tf.placeholder(tf.float32, shape=[None, 784])  #[batchsize, 784]

y = tf.placeholder(tf.float32, shape=[None, 10])   #[batchsize, 10]

phase_train = tf.placeholder(tf.bool)

keep_prob = tf.placeholder(tf.float32)
#you can skip this till model is fully created in below cells

def batchnorm_layer(logits, n_out, conv=True):

    """ Apply batch normalization before feeding into activation function. 

    During train phase simply calculate means & variance of given batch as well as keeps an

    moving average of these means and variances for all the batches to be used during training

    time.

    Args:

        logits: values before activation fun. If conv==True, dimension is 

        [batch, height, width, n_out] else [batch, n_out]

        n_out: size of mean and variance vector

        conv: Default True, if batch norm is applied to output of conv layer

    Return:

        outputs after applying batch normalization

    """

    offset = tf.Variable(tf.constant(0.0, shape=[n_out]))

    scale = tf.Variable(tf.constant(1.0, shape=[n_out]))

        

    exp_moving_avg = tf.train.ExponentialMovingAverage(0.99)

    if conv:

        mean, variance = tf.nn.moments(logits, [0, 1, 2])

    else:

        mean, variance = tf.nn.moments(logits, [0])



    def mean_var_with_update():

        update_moving_avg = exp_moving_avg.apply([mean, variance])

        with tf.control_dependencies([update_moving_avg]):

            return tf.identity(mean), tf.identity(variance)



    m, v = tf.cond(phase_train, mean_var_with_update\

                   , lambda: (exp_moving_avg.average(mean), exp_moving_avg.average(variance)))



    Ybn = tf.nn.batch_normalization(logits, m, v, offset, scale, variance_epsilon=1e-5)

    return Ybn
#layer 1

#filter size = [5,5,1] = [height, width, input_channels] and 32 such filters i.e output_channels

#stride = [1,1] with same padding

w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))

b1 = tf.Variable(tf.constant(0.1, shape=[32])) #non zero bias

x_image = tf.reshape(x, [-1, 28, 28, 1])                        #[batch_size, 28, 28, 1]                      



conv1 = tf.nn.conv2d(x_image, w1, [1, 1, 1, 1], padding="SAME") #[batch_size, 28, 28, 32]

bn1 = batchnorm_layer(tf.add(conv1, b1), 32)                    #[batch_size, 28, 28, 32]

#a1 = tf.nn.relu(tf.add(conv1, b1))

a1 = tf.nn.relu(bn1)                                            #[batch_size, 28, 28, 32]



#max pooling filter size = [2,2]

#max pooling stride = [2,2] with same padding

pool1 = tf.nn.max_pool(a1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")  #[batch_size, 14, 14, 32]
#layer 2



w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))

b2 = tf.Variable(tf.constant(0.1, shape=[64]))



conv2 = tf.nn.conv2d(pool1, w2, [1, 1, 1, 1], padding="SAME")  #[batch_size, 14, 14, 64]

bn2 = batchnorm_layer(tf.add(conv2, b2), 64)

#a2 = tf.nn.relu(tf.add(conv2, b2))

a2 = tf.nn.relu(bn2)



pool2 = tf.nn.max_pool(a2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME") #[batch_size, 7, 7, 64]
#fully connectec layer



w3 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))

b3 = tf.Variable(tf.constant(0.1, shape=[1024]))



flat1 = tf.reshape(pool2, [-1, 7 * 7 * 64])                    #[batch_size, 3136]

fc1 = tf.add(tf.matmul(flat1, w3), b3)                         #[batch_size, 1024]

#a3 = tf.nn.relu(fc1)

bn3 = batchnorm_layer(fc1, 1024, conv=False)

a3 = tf.nn.relu(bn3)                                           #[batch_size, 1024]
#dropout for regularization



dropout = tf.nn.dropout(a3, keep_prob=keep_prob)
#output layer



w4 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))

b4 = tf.Variable(tf.constant(0.1, shape=[10]))



pred = tf.matmul(dropout, w4) + b4                             #[batch_size, 10]
global_step = tf.Variable(0, trainable=False)



learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, int(samples/batch_size), 0.96, staircase=True)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
inn = tf.global_variables_initializer()

sess = tf.InteractiveSession()



sess.run(inn)



for epoch in range(epochs):



    avg_cost = 0.0

    total_batch = int(samples/batch_size)

    

    for i in range(total_batch):

        batch_x, batch_y = X[i*batch_size:(i+1)*batch_size], Y[i*batch_size:(i+1)*batch_size]

        

        #because of batch normalization which also helps in regularization using less max pooling

        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: .20, phase_train: True})

        avg_cost += c / total_batch

    print("Epoch: {} cost={:.4f}".format(epoch+1,avg_cost))

print("Model has completed {} Epochs of Training".format(epochs))
test_images = pd.read_csv('test.csv')

X_test = test_images.as_matrix()



#no max pooling during test time

w_value = sess.run(pred,feed_dict={x: X_test, keep_prob: 1.0, phase_train: False})
output = pd.DataFrame(w_value.argmax(axis=1), columns=["Label"],index=range(1,28001))

output.index.names = ['ImageId']

output.to_csv('outputConvBN.csv')