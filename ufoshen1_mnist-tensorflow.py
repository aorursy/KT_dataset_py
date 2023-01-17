import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
%matplotlib inline

# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))
# load train.csv
trainDf = pd.read_csv("../input/train.csv")
trainDf.describe()
# load test.csv
testDf = pd.read_csv("../input/test.csv")
testDf.describe()
# make train set, dev set, and test set, and see pictures
trainDf = shuffle(trainDf)                              # shuffle
trainXYOrig = trainDf.values
m = trainXYOrig.shape[0]
partition = int(m * 0.8)
trainXOrig = trainXYOrig[:, 1:].reshape(-1, 28, 28, 1)  # m * height * width * 1
trainX = trainXOrig / 255                               # 0-1
devX = trainX[partition: , :, :, :]
trainX = trainX[0: partition, :, :, :]

trainYOrig = trainXYOrig[:, 0]
trainY = np.eye(10)[trainYOrig, :]                      # one-hot
devY = trainY[partition: , :]
trainY = trainY[0: partition, :]

testXOrig = testDf.values.reshape(-1, 28, 28, 1)
testX = testXOrig / 255

print("trainX.shape = "  + str(trainX.shape))
print("trainY.shape = "  + str(trainY.shape))
print("devX.shape = "  + str(devX.shape))
print("devY.shape = "  + str(devY.shape))
print("testX.shape = "  + str(testX.shape))

# Visualize one of the images
index = 39
plt.imshow(trainX[index, :, :, 0], cmap ='gray')
plt.title("Train Set Image "+ str(index) + ": label = " + str(trainYOrig[index]))
# function random_mini_batches: make minibatches of X and Y
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    num_complete_minibatches = int(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: (k + 1) * mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: (k + 1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
# X Y: placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
is_training = tf.placeholder(tf.bool)
print("X = " + str(X))
print("Y = " + str(Y))
# W1, W2：tensorflow variable, use Xavier initialization
W1 = tf.get_variable("W1", shape=[3, 3, 1, 4], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[3, 3, 4, 8], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[3, 3, 8, 8], initializer=tf.contrib.layers.xavier_initializer()) # 加一层
# Layer 1: CONV2D(Z1) -> BatchNorm(N1) -> RELU(A1) -> MAXPOOL(P1) -> 
Z1 = tf.nn.conv2d(input=X, filter=W1, strides=[1, 1, 1, 1], padding='SAME')
N1 = tf.layers.batch_normalization(Z1, training=is_training)                        # batch norm should be applied in "layers" dimension，here: layers_last，use default axis = -1
A1 = tf.nn.relu(N1)
P1 = tf.nn.max_pool(value=A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# layer 2: CONV2D(Z2) -> BatchNorm(N2) -> RELU(A2) -> MAXPOOL(P2) -> 
Z2 = tf.nn.conv2d(input=P1, filter=W2, strides=[1, 1, 1, 1], padding='SAME')
N2 = tf.layers.batch_normalization(Z2, training=is_training)
A2 = tf.nn.relu(N2)
P2 = tf.nn.max_pool(value=A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# layer 3: CONV2D(Z3) -> BatchNorm(N3) -> RELU(A3) -> MAXPOOL(P3) -> 
Z3 = tf.nn.conv2d(input=P2, filter=W3, strides=[1, 1, 1, 1], padding='SAME')
N3 = tf.layers.batch_normalization(Z3, training=is_training)
A3 = tf.nn.relu(N3)
P3 = tf.nn.max_pool(value=A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# layer 4: FLATTEN(F3) -> FULLYCONNECTED(Z4) -> BatchNorm(N4) -> RELU(A4)
F3 = tf.contrib.layers.flatten(P3)
Z4 = tf.contrib.layers.fully_connected(F3, num_outputs=64, activation_fn=None)
N4 = tf.layers.batch_normalization(Z4, training=is_training)
A4 = tf.nn.relu(N4)

# layer 5: FULLYCONNECTED(Z5)
Z5 = tf.contrib.layers.fully_connected(A4, num_outputs=10, activation_fn=None)

# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5, labels=Y))
# hyperparameters
learning_rate = 0.002
num_epochs = 50
mini_batch_size = 64
costs = []
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()                                      # 要在 Adam 之后再来initializer,否则beta1, beta2没有initialize
sess = tf.Session()
sess.run(init)
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)       # for batch norm
for epoch in range(num_epochs):
    minibatches = random_mini_batches(trainX, trainY, mini_batch_size=mini_batch_size, seed=epoch + int(time.time()))
    epoch_cost = 0.

    for minibatch in minibatches:
        (minibatchX, minibatchY) = minibatch
        _, mini_batch_cost, __ = sess.run([optimizer, cost, extra_update_ops], feed_dict={X: minibatchX, Y: minibatchY, is_training:True})
        epoch_cost += mini_batch_cost

    # print and save costs
    epoch_cost /= len(minibatches)
    costs.append(epoch_cost)
    if epoch % 2 == 0:
        print("No. %d epoch, cost = %f" % (epoch, epoch_cost))
        
plt.plot(costs)
plt.xlabel("number of epochs")
plt.ylabel("cost")
plt.title("cost")
# computational graph
predict_op = tf.argmax(Z5, 1)
correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
train_accuracy = accuracy.eval(session=sess, feed_dict={X: trainX, Y: trainY, is_training:False})
dev_accuracy = accuracy.eval(session=sess, feed_dict={X: devX, Y: devY, is_training:False})
print("train_accuracy = " + str(train_accuracy))
print("dev_accuracy = " + str(dev_accuracy))
testY_pred = predict_op.eval(session=sess, feed_dict={X: testX, is_training:False})
testYDf = pd.DataFrame(testY_pred.reshape(-1, 1), index=np.arange(1, 1 + len(testY_pred)), columns=["Label"]) # index 要求从 1 开始
testYDf.to_csv("test_predict.csv", index=True, index_label="ImageId")
