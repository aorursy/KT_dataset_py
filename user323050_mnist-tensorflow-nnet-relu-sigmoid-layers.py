import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv
import math
def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def load_dataset():
    df = pd.read_csv('../input/train.csv', header=0)
    df_array = np.array(df)
    images = df_array[:, 1:].reshape(df_array.shape[0], 784).astype('float32') / 255.0
    labels = np.array(pd.get_dummies(df['label']))

    return images, labels

def load_test_dataset():
    df_test = pd.read_csv('../input/test.csv', header=0)
    df_array_test = np.array(df_test)
    images_test = df_array_test.reshape(df_array_test.shape[0], 28 * 28).astype('float32') / 255.0
    return images_test

def initialize_parameters():
    W_relu = tf.get_variable("W_relu", [784, 100], initializer = tf.contrib.layers.xavier_initializer())
    b_relu = tf.get_variable("b_relu", [100], initializer = tf.zeros_initializer())
    W = tf.get_variable("W", [100, 10], initializer = tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [10], initializer = tf.zeros_initializer())
    parameters = {
        "W_relu": W_relu,
        "b_relu": b_relu,
        "W": W,
        "b": b
    }
    return parameters

def forward_propagation(X, parameters, keep_probability):
    W_relu = parameters['W_relu']
    b_relu = parameters['b_relu']
    W = parameters['W']
    b = parameters['b']

    h = tf.nn.relu(tf.matmul(X, W_relu) + b_relu)
    h_drop = tf.nn.dropout(h, keep_probability)

    logits = tf.matmul(h_drop, W) + b
    return logits

def compute_cost(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

def train_model(X_train, X_test, Y_train, Y_test, num_epochs=200, minibatch_size=500, learning_rate=0.001):
    m = X_train.shape[0]

    keep_probability = tf.placeholder(tf.float32)

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    parameters = initialize_parameters()
    logits = forward_propagation(X, parameters, keep_probability)
    cost = compute_cost(logits, Y)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0.0
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                batch_xs, batch_ys = minibatch
                _, batch_cost = sess.run([optimizer, cost], feed_dict = {
                    X: batch_xs,
                    Y: batch_ys,
                    keep_probability: 0.5
                })
                epoch_cost += np.sum(batch_cost / minibatch_size)

            if epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

        parameters = sess.run(parameters)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("Точность: %s" % sess.run(accuracy, feed_dict={X: X_test, Y: Y_test, keep_probability: 1.}))

        # saver.save(sess, './mnist-checkpoint/model.ckpt')

        return parameters

def predict(X, parameters):
    W_relu = tf.convert_to_tensor(parameters['W_relu'])
    b_relu = tf.convert_to_tensor(parameters['b_relu'])
    W = tf.convert_to_tensor(parameters['W'])
    b = tf.convert_to_tensor(parameters['b'])

    x = tf.placeholder(tf.float32, [None, 784])
    h = tf.nn.relu(tf.matmul(x, W_relu) + b_relu)
    logits = tf.matmul(h, W) + b

    p = tf.argmax(logits, 1)

    with tf.Session() as sess:
        prediction = sess.run(p, feed_dict = {x: X})
        return prediction

images, labels = load_dataset()
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.2)
parameters = train_model(X_train, X_test, Y_train, Y_test, num_epochs=1500)
images_test = load_test_dataset()
predictions = predict(images_test, parameters)
predictions = [[i+1, number] for i, number in enumerate(predictions)]
submission = pd.DataFrame(predictions, columns=['ImageId', 'Label'])
submission.to_csv('submission.csv', index=False)
