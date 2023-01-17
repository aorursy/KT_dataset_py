import math

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

plt.style.use('seaborn')

# Load data into mnist.train (60K samples) and mnist.test (10K samples)
mnist = input_data.read_data_sets('/tmp/MNIST_data', one_hot=True,
                                  reshape=False, validation_size=False)
# Input data
# None corresponds to the the number of images in the mini-batch. It will be
# known at training time
X = tf.placeholder(tf.float32, [None, 28, 28, 1])  # images 28x28x1 (grayscale)
Y = tf.placeholder(tf.float32, [None, 10])  # true label

# Transform the 28x28 images into single vectors of 784 pixels
XX = tf.reshape(X, [-1, 784])  # -1 works similar to np.reshape()
# Architecture for neural network with 5 layers
#
# · · · · · · · · · · (input data, flattened pixels)   X [batch, 784]
# \x/x\x/x\x/x\x/x\x/ fully connected layer (relu)     W1 [784, 200]    B1[200]
#  · · · · · · · · ·                                   Y1 [batch, 200]
#   \x/x\x/x\x/x\x/   fully connected layer (relu)     W2 [200, 100]    B2[100]
#    · · · · · · ·                                     Y2 [batch, 100]
#     \x/x\x/x\x/     fully connected layer (relu)     W3 [100, 60]     B3[60]
#      · · · · ·                                       Y3 [batch, 60]
#       \x/x\x/       fully connected layer (relu)     W4 [60, 30]      B4[30]
#        · · ·                                         Y4 [batch, 30]
#         \x/         fully connected layer (softmax)  W5 [30, 10]      B5[10]
#          ·                                           Y5 [batch, 10]

# Dimensions for our intermediary layers
L = 200
M = 100
N = 60
O = 30

# Earlier we've initialized weights with tf.zeros(). But it's a good practice
# to use random values instead
# When using ReLUs, another good practice is to initialize biases with small
# positive values. For example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 28x28
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# The last layer will continue using softmax, because it's better for prediction
# The intermediary layers, on the other hand, will use a ReLU
# The sigmoid is also a classical candidate, but it's actually quite problematic
# in deep networks. It squashes all values between 0 and 1 and when you do so
# repeatedly, neuron outputs and their gradients can vanish entirely
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y_pred = tf.nn.softmax(Ylogits)
# Define the loss function
# Previously we've used our own custom cross entropy function, but TensorFlow
# has a handy function implemented in a numerically stable way, e.g.: for log(0)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# Accuracy of the trained model
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Define the optimizer and ask it to minimize the cross entropy loss
# Previously we've used the GradientDescentOptimizer, but AdamOptimizer is
# more robust to "saddle points" of the gradient
# We've also used a fixed learning rate of 0.003, which was too high. The
# decay will decrease the learning rate exponentially from 0.003 to 0.0001
step = tf.placeholder(tf.int32)
lrate = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
optimizer = tf.train.AdamOptimizer(lrate)
train_step = optimizer.minimize(cross_entropy)
# Initialize the variables and the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# Scores data that we're going to plot later
train_scores = dict(acc=[], loss=[])
test_scores = dict(acc=[], loss=[])
# The main loop of training
# We can use a dictionary to feed the actual data into the placeholders
iterations = 10001
print_freq = iterations//10
for i in range(iterations):
    # Load batch of images and true labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    a, c = sess.run([accuracy, cross_entropy],
                    feed_dict={X: batch_X, Y: batch_Y})
    train_scores['acc'].append(a)
    train_scores['loss'].append(c)

    if i % print_freq == 0:
        print(f'{i:4d}: train acc {a:.4f}, train loss {c:.5f}')

    a, c = sess.run([accuracy, cross_entropy],
                    feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    test_scores['acc'].append(a)
    test_scores['loss'].append(c)

    # Train
    # The iteration number will be used to decay the learning rate
    sess.run(train_step, feed_dict={X: batch_X, Y: batch_Y, step: i})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(train_scores['acc'], lw=0.5)
ax1.plot(test_scores['acc'])
ax1.set_title('Accuracy')
ax1.set_ylim([0.94, 1.0])

ax2.plot(train_scores['loss'], lw=0.5)
ax2.plot(test_scores['loss'])
ax2.set_title('Cross entropy loss')
ax2.set_ylim([0, 20])

fig.legend(labels=('train', 'test'), loc='lower center',
           ncol=2, frameon=True, fontsize='medium')
plt.show()
# Check the accuracy in the test set
a, c = sess.run([accuracy, cross_entropy],
                feed_dict={X: mnist.test.images, Y: mnist.test.labels})
print(f'Test acc {a:.4f}, test loss {c:.5f}')
# Release the resources when they are no longer required
# It's also possible to use tf.Session() as a context manager instead
sess.close()