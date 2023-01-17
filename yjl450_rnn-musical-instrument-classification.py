import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

# from tensorflow.python.ops import rnn, rnn_cell

from tensorflow.contrib import rnn

from tensorflow.keras import layers

import pickle
# Load training data

data = np.load('../input/irmastraining/sample.npy')

labels = np.load('../input/irmastraining/label.npy').astype(int)

print(data[0][0])

print(labels.shape)



# Load testing data

file = open('../input/test1irmas/test1.pickle', 'rb')

test = pickle.load(file)

file.close

test_data = test['sample']

test_label = np.array(test['label'])

print(test_data[0].shape)

print(test_label.shape)

print('Different Instruments: ',end = '')

cata = [0]*10

for i in test_label:

    cata[i] += 1

print(cata)



# Define softmax

def cal_softmax(list):

    input_array = list[0]

    output = np.zeros(len(input_array))

    total = 0

    for i in range(len(input_array)):

        total += np.exp(input_array[i])

    for i in range(len(input_array)):

        output[i] = np.exp(input_array[i]) / total

    return output



print(cal_softmax(np.array([[1,2,3,4,5,6,7,8,9,10]])))
def one_hot_encode(labels):

    n_labels = labels.shape[0]

    n_unique_labels = len(np.unique(labels))

    one_hot_encode = np.zeros((n_labels,10))

    one_hot_encode[np.arange(n_labels), labels] = 1

    return one_hot_encode

one_hot_labels = one_hot_encode(labels)

print(one_hot_labels.shape)

print(one_hot_labels)

one_hot_test = one_hot_encode(test_label)

print(one_hot_test)
tf.reset_default_graph()



learning_rate = 0.001

training_iters = 100

batch_size = 128

display_step = 10



# Network Parameters

n_input = 40

n_steps = 130

n_hidden = 100

n_classes = 10 



x = tf.placeholder("float", [None, n_steps, n_input])

y = tf.placeholder("float", [None, n_classes])



weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))

bias = tf.Variable(tf.random_normal([n_classes]))
def RNN(x, weight, bias):

    cell = layers.LSTMCell(n_hidden)

    output, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)

    output = tf.transpose(output, [1, 0, 2])

    last = tf.gather(output, int(output.get_shape()[0]) - 1)

    

    return tf.matmul(last, weight) + bias

# def RNN(x, weights, biases):

#     x = tf.unstack(x, n_steps, 1)

#     lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

#     outputs, states = rnn.Dynamic_rnn(lstm_cell, x, dtype=tf.float32)

#     return tf.matmul(outputs[-1], weights['out']) + biases['out']
prediction = RNN(x, weight, bias)



# Define loss and optimizer

# loss_f = -tf.reduce_sum(y * tf.log(prediction))

loss_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_f)



# Evaluate model

correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# Initializing the variables

init = tf.global_variables_initializer()



# Save the accuracy for future graph

loss_graph = []

accuracy_graph = []

iter_axis = []
with tf.Session() as session:

    session.run(init)

    

    for itr in range(training_iters):

        iter_loss = 0

        iter_acc = 0

        for k in range(5927//batch_size):

            offset = (k * batch_size) % (data.shape[0] - batch_size)

            batch_x = data[offset:(offset + batch_size), :, :]

            batch_y = one_hot_labels[offset:(offset + batch_size), :]

            _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})

            # compute loss and acc for that minibatch

            # add that batch loss to iter_loss

            acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})

            loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})

            iter_acc += acc

            iter_loss += loss

        iter_acc /= 5927//batch_size

        iter_loss /= 5927//batch_size

        print("Iter " + str(itr) + ", Loss= " + "{:.6f}".format(iter_loss) + ", Training Accuracy=" + "{:.5f}".format(iter_acc))

        iter_axis.append(itr)

        loss_graph.append(iter_loss)

        accuracy_graph.append(iter_acc)

    training = {'loss': loss_graph, 'iter':iter_axis, 'accu':accuracy_graph}

    file = open('data.pkl', 'wb')

    pickle.dump(training, file)

    file.close()

    print('Start testing')

    # Test data

    test_correct=[0]*11

    for case in range(len(test_data)):

        print(case)

        total_pred = np.zeros((1,10))

        for i in range(len(test_data[case])//130):

            data_point = test_data[case][i*130: i*130+130][np.newaxis, :]

            total_pred += session.run(prediction, feed_dict={x: data_point})

        soft_max = cal_softmax(total_pred)

        pred = np.argmax(soft_max)

        data_label = test_label[case]

        if pred == data_label:

            print('Correct Pred')

            test_correct[10] += 1

            test_correct[data_label] += 1

            print(test_correct)

    print(test_correct)
# Plot the accuracy

plt.xlabel("Iteration") 

plt.ylabel("Average Accuracy") 

plt.plot(iter_axis, accuracy_graph)

plt.show()
# Plot the loss

plt.xlabel("Iteration") 

plt.ylabel("Average Loss") 

plt.plot(iter_axis, loss_graph)

plt.show()