import tensorflow as tf

from tqdm import tqdm

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
# Checking the train dataframe

train_data.head()
train_data.info()
# Separating the Input value as X_train and output value as Y_train

Y_train = train_data["label"]

X_train = train_data.drop("label", axis=1)
# Plotting the frequency of different numbers 

plt.figure(figsize=(10,5))

sns.set_style("dark")

sns.countplot(x=Y_train)
X_train.head()
X_train.head().values.shape
Y_train.head(10)
# Showing the output of One Hot Encoding of output values

pd.get_dummies(Y_train.head(25))
Y_train = pd.get_dummies(Y_train)
# Output value after one hot encoded

Y_train.head()
Y_train.head().values.shape
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
total = int(train_data.shape[0]/100)

total
epochs = 100

for i in tqdm(range(epochs)):

    for j in range(total):

        batch_x = X_train.iloc[i*100:(i+1)*100].values

        batch_y = Y_train.iloc[i*100:(i+1)*100].values



        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
# Accuracy on training data

matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

acc = tf.reduce_mean(tf.cast(matches,tf.float32))

print(sess.run(acc*100,feed_dict={x:X_train.values,y_true:Y_train.values,hold_prob:1.0}))
saver = tf.train.Saver()
saver.save(sess, "./Digit Model/")
# saver.restore(sess, "./input/Digit Model/")
test_data.head()
test_data.info()
result = sess.run(tf.math.argmax(y_pred,1),feed_dict={x:test_data.values,hold_prob:1.0})
result.shape
result = result.reshape(-1,1)
result.shape
final_result = pd.DataFrame(result, columns=["Label"],index=np.arange(1,28001))
final_result.head()
final_result.reset_index(inplace=True)
final_result.head()
final_result.columns = ["ImageId", "Label"]
final_result.head()
# Final submission file

final_result.to_csv("digit_recognition_submission.csv",index=False)