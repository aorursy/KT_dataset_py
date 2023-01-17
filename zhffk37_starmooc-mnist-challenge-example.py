import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.cm as cm  
def make_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    one_hot = np.zeros((num_labels, num_classes))
    one_hot.flat[index_offset + labels.ravel()] = 1
    return one_hot
train = pd.read_csv('../input/train.csv') 
test = pd.read_csv('../input/test.csv')

x_train = train.iloc[:,1:]
y_train = make_one_hot(train.iloc[:,0],10)

x_test = test.iloc[:,0:]
for i in range(10):
    index = np.where(train['label']==i)[0][0]
    plt.imshow(np.array(x_train.iloc[index]).reshape(28,28),cmap=cm.binary)
    plt.show()
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

W1 = tf.Variable(weight_variable((784,100)))
b1 = tf.Variable(tf.zeros([100]))
W2 = tf.Variable(weight_variable((100,10)))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

h1 = y = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h1, W2))
learning_rate = 0.5
num_of_epoch = 100
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in tqdm(range(num_of_epoch)):
    X = x_train
    Y = y_train
    sess.run(train_step, feed_dict={x: X, y_: Y})
# Define accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Train Accuracy = %f'%sess.run(accuracy, feed_dict={x: x_train, y_: y_train}))
# Define predict
predict_y = tf.argmax(y,1)
predict_label = sess.run(predict_y,feed_dict={x:x_test})
result = pd.DataFrame({'Label':predict_label})
result.index.name = 'ImageId'
result.index = result.index+1
result.to_csv('submission.csv')