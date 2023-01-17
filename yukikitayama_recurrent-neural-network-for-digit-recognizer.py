import pandas as pd

import numpy as np



import tensorflow as tf



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder



from tensorflow.keras.utils import to_categorical
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
train.shape
train.head()
test.head()
y = train['label'].values.reshape((-1, 1))

X = train.drop('label', axis = 1).values

print(y.shape, X.shape)
y_oh = to_categorical(y)

y_oh.shape
y_oh
X_train, X_vali, y_train, y_vali = train_test_split(X, y_oh, test_size = 0.2, random_state = 42)
print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
tf.reset_default_graph()
# parameters

element_size = 28

time_steps = 28

num_classes = 10

batch_size = 280

hidden_layer_size = 256
x = tf.placeholder(tf.float32, shape = [None, time_steps, element_size], name = 'inputs')

y = tf.placeholder(tf.float32, shape = [None, num_classes], name = 'labels')
validation_data = X_vali[:batch_size].reshape((-1, time_steps, element_size))

validation_label = y_vali[:batch_size]
def next_batch(num, data, labels):

    idx = np.arange(0, len(data))

    np.random.shuffle(idx)

    idx = idx[:num]

    data_shuffle = [data[i] for i in idx]

    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))

Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))

b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))



def rnn_step(previous_hidden_state, x):

    current_hidden_state = tf.tanh(tf.matmul(previous_hidden_state, Wh) + tf.matmul(x, Wx) + b_rnn)

    return current_hidden_state    



processed_input = tf.transpose(x, perm = [1, 0, 2])

initial_hidden = tf.zeros([batch_size, hidden_layer_size])



all_hidden_states = tf.scan(rnn_step, processed_input, initializer = initial_hidden)



Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean = 0, stddev = 0.01))

bl = tf.Variable(tf.truncated_normal([num_classes], mean = 0, stddev = 0.01))



def get_linear_layer(hidden_state):

    return tf.matmul(hidden_state, Wl) + bl



all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)

output = all_outputs[-1]

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y))

train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))

accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())



for i in range(10000):

    

    batch_x, batch_y = next_batch(batch_size, X_train, y_train)

    batch_x = batch_x.reshape((batch_size, time_steps, element_size))

    sess.run(train_step, feed_dict = {x: batch_x, y: batch_y})

    

    if i % 1000 == 0:

        acc, loss = sess.run([accuracy, cross_entropy], feed_dict = {x: batch_x, y: batch_y})

        val_acc = sess.run(accuracy, feed_dict = {x: validation_data, y: validation_label})

        print("Iteration = " + str(i) + " Loss = {:.6f}".format(loss) + " Accuracy = {:.5f}".format(acc) +

              " Validation accuracy = {:.5f}".format(val_acc))
X_test = test.values.reshape((-1, time_steps, element_size))

print(test.shape)

print(X_test.shape)
y_pred_list = []



for i in range(X_test.shape[0]//batch_size):

    X_test_iter = X_test[(i*batch_size):((i+1)*batch_size)]

    y_pred = sess.run(tf.argmax(output, 1), feed_dict = {x: X_test_iter})

    

    y_pred_list.append(y_pred)

    

y_pred_list = [item for sublist in y_pred_list for item in sublist]
test_id = np.arange(1, X_test.shape[0] + 1, 1)

test_id
print(test_id.shape)

print(len(y_pred_list))
sub = pd.DataFrame(data = {'ImageId': test_id,

                           'Label': y_pred_list})

sub.head()