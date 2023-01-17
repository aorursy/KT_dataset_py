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
# Parameters the network is going to learn
W = tf.Variable(tf.zeros([784, 10]))  # 28x28
b = tf.Variable(tf.zeros([10]))

# Single 1-layer neural network
# This will be the final prediction
Y_pred = tf.nn.softmax(tf.matmul(XX, W) + b)
# Define the loss function
# In a real training use tf.nn.softmax_cross_entropy_with_logits() instead
# Although the original formula uses a sum, this trick allows to plot the train
# and test data using the same scale. It's 1000 because we're using batches of
# 100 images and *10 because mean included an unwanted divison by 10
cross_entropy = -tf.reduce_mean(Y * tf.log(Y_pred)) * 1000

# Accuracy of the trained model
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Define the optimizer and ask it to minimize the cross entropy loss
learning_rate = 0.003
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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
iterations = 2001
print_freq = iterations//10
for i in range(iterations):
    # Load batch of images and true labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y: batch_Y}

    # Train
    sess.run(train_step, feed_dict=train_data)

    # Train accuracy and loss
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    train_scores['acc'].append(a)
    train_scores['loss'].append(c)

    if i % print_freq == 0:
        print(f'{i:4d}: train acc {a:.4f}, train loss {c:.5f}')

    # Test accuracy and loss
    test_data = {X: mnist.test.images, Y: mnist.test.labels}
    a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    test_scores['acc'].append(a)
    test_scores['loss'].append(c)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(train_scores['acc'], label='train', lw=0.5)
ax1.plot(test_scores['acc'], label='test')
ax1.set_title('Accuracy')
ax1.legend()

ax2.plot(train_scores['loss'], label='train', lw=0.5)
ax2.plot(test_scores['loss'], label='test')
ax2.set_title('Cross entropy loss')
ax2.set_ylim([0, 100])
ax2.legend()

plt.show()
# Check the accuracy in the test set
test_data = {X: mnist.test.images, Y: mnist.test.labels}
a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
print(f'Test acc {a:.4f}, test loss {c:.5f}')
# Release the resources when they are no longer required
# It's also possible to use tf.Session() as a context manager instead
sess.close()