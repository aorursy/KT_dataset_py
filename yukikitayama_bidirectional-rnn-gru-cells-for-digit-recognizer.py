import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
y = train['label'].values.reshape((-1, 1))

X = train.drop('label', axis = 1).values

y_oh = to_categorical(y)

print(y.shape)

print(y_oh.shape)
X_train, X_vali, y_train, y_vali = train_test_split(X, y_oh, test_size = 0.2, random_state = 42)

print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
tf.reset_default_graph()
def next_batch(num, data, labels):

    idx = np.arange(0, len(data))

    np.random.shuffle(idx)

    idx = idx[:num]

    data_shuffle = [data[i] for i in idx]

    labels_shuffle = [labels[i] for i in idx]

    

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
# parameters

time_steps = 784

batch_size = 280

num_classes = 10

hidden_layer_size = 16
_inputs = tf.placeholder(tf.float32, shape = [batch_size, time_steps, 1])

_labels = tf.placeholder(tf.float32, shape = [batch_size, num_classes])
validation_data = X_vali[:batch_size].reshape((-1, time_steps, 1))

validation_label = y_vali[:batch_size]
print(validation_data.shape)

print(validation_label.shape)
with tf.name_scope("biGRU"):

    with tf.variable_scope('forward'):

        gru_fw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)

        gru_fw_cell = tf.contrib.rnn.DropoutWrapper(gru_fw_cell)



    with tf.variable_scope('backward'):

        gru_bw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)

        gru_bw_cell = tf.contrib.rnn.DropoutWrapper(gru_bw_cell)



    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw = gru_fw_cell,

                                                      cell_bw = gru_bw_cell,

                                                      inputs = _inputs,

                                                      dtype = tf.float32,

                                                      scope = "BiGRU")



states = tf.concat(values = states, axis = 1)
weights = {'linear_layer': tf.Variable(tf.truncated_normal([2 * hidden_layer_size, num_classes], mean = 0, stddev = 0.01))}

biases = {'linear_layer': tf.Variable(tf.truncated_normal([num_classes], mean = 0, stddev = 0.01))}



final_output = tf.matmul(states, weights["linear_layer"]) + biases["linear_layer"]

softmax = tf.nn.softmax_cross_entropy_with_logits(logits = final_output, labels = _labels)

cross_entropy = tf.reduce_mean(softmax)



train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(_labels, 1), tf.argmax(final_output, 1))

accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())



for i in range(100):

    batch_x, batch_y = next_batch(batch_size, X_train, y_train)

    batch_x = batch_x.reshape((batch_size, time_steps, 1))

    sess.run(train_step, feed_dict = {_inputs: batch_x,

                                      _labels: batch_y})

    

    if i % 10 == 0:

        acc, loss = sess.run([accuracy, cross_entropy], feed_dict = {_inputs: batch_x,

                                                                     _labels: batch_y})

        val_acc = sess.run(accuracy, feed_dict = {_inputs: validation_data,

                                                  _labels: validation_label})

        print("Iter = " + str(i) + " Loss = {:.6f}".format(loss) + " Accuracy = {:.5f}".format(acc) + 

              " Validation accuracy = {:.5f}".format(val_acc))
X_test = test.values.reshape((-1, time_steps, 1))

print(test.shape)

print(X_test.shape)
y_pred_list = []



for i in range(X_test.shape[0]//batch_size):

    X_test_iter = X_test[(i*batch_size):((i+1)*batch_size)]

    y_pred = sess.run(tf.argmax(final_output, 1), feed_dict = {_inputs: X_test_iter})

    

    y_pred_list.append(y_pred)

    

y_pred_list = [item for sublist in y_pred_list for item in sublist]
test_id = np.arange(1, X_test.shape[0] + 1, 1)

print(test_id.shape)

print(len(y_pred_list))
sub = pd.DataFrame(data = {'ImageId': test_id,

                           'Label': y_pred_list})

sub.head()