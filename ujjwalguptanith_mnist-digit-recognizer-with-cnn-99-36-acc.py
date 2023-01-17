import tensorflow as tf

from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../input/mnist_data/MNIST_data", one_hot = True)
def init_weights(shape):

    init_random_dist = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(init_random_dist)
def init_bias(shape):

    init_random_bias = tf.constant(0.1, shape=shape)

    return tf.Variable(init_random_bias)
def conv2d(x,W):

    # Creating a convolutional Neural Network

    # x ----> [batch, H, W, Channel]

    # W ----> [filter H, filter w, channels IN, channel out]

    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")
def max_pool_2by2(x):

    # function for max pooling

    # x ----> [batch, H, W, Channel]

    return tf.nn.max_pool(x, ksize=[1,2,2,1] , strides = [1,2,2,1], padding="SAME")
def convolutional_layer(input_x, shape):

    # returning the output after passing the Convolutional Neural Network to Relu Activation Function

    W = init_weights(shape)

    b = init_bias([shape[3]])

    return tf.nn.relu(conv2d(input_x, W) + b)
def normal_full_layer(input_layer, size):

    # Fully connected layer for last

    input_size = int(input_layer.get_shape()[1])

    W = init_weights([input_size, size])

    b = init_bias([size])

    return tf.matmul(input_layer, W) + b
# Placeholders for our input x and output y

x = tf.placeholder(tf.float32, shape=[None, 784])

y_true = tf.placeholder(tf.float32, shape=[None,10])
# Reshaping our input x (2-D) into our accepted input for CNN Network that is 4-D

x_image = tf.reshape(x, shape=[-1,28,28,1]) # 784 = 28*28
convo1 = convolutional_layer(x_image, shape = [5,5,1,32])

convo_1_pooling = max_pool_2by2(convo1)
convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])

convo_2_pooling = max_pool_2by2(convo_2)
# After first max pooling : 28/2 = 14

# After second max pooling : 14/2 = 7

convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])

full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

#1024 is the nos. of neurons we want in our fully connected layer
hold_prob = tf.placeholder(tf.float32)

full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
y_pred = normal_full_layer(full_one_dropout,10)
print("input Size: ", x.get_shape())

print("After reshaping, input Size: ", x_image.get_shape())

print("After first conolution: ", convo1.get_shape())

print("After first Pooling: ", convo_1_pooling.get_shape())

print("After second conolution: ", convo_2.get_shape())

print("After second Pooling: ", convo_2_pooling.get_shape())

print("After flatening: ",convo_2_flat.get_shape())

print("After first fully dense NN: ",full_layer_one.get_shape())

print("After first dropout: ",full_one_dropout.get_shape())

print("Prediction: ", y_pred.get_shape())
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
epochs = 50000

for i in tqdm(range(epochs)):



    batch_x , batch_y = mnist.train.next_batch(50)



    sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
saver = tf.train.Saver()
# To restore the previous weights run the next command

saver.save(sess, "./CNN Model/CNN Model")
saver.restore(sess, "./CNN Model/CNN Model")
matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

acc = tf.reduce_mean(tf.cast(matches,tf.float32))

print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}))
sess.close()