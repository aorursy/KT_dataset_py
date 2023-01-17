import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
np.random.seed(42)
X_train = np.linspace(0, 50, 200) + np.random.normal(0, 3, 200)
y_train = np.linspace(0, 50, 200) + np.random.normal(0, 3, 200)

plt.scatter(X_train, y_train);
import tensorflow as tf

X = tf.placeholder('float32')
y = tf.placeholder('float32')

w0 = tf.Variable(np.random.random())
w1 = tf.Variable(np.random.random())
y_pred = tf.add(tf.multiply(w1, X), w0)
cost = tf.reduce_mean(tf.square(tf.subtract(y, y_pred)))
learning_rate = 0.0001
epochs = 1000
display_step = 100

# Note, minimize() knows to modify w0 and w1 because Variable objects are
# trainable=True by default
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for e in range(epochs):
        sess.run(train_step, feed_dict={X: X_train, y: y_train})

        if (e + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: X_train, y: y_train})
            print('Epoch: {:4d} cost = {:.9f} w0 = {:.6f} w1 = {:.6f}'.format(
                e + 1, c, sess.run(w0), sess.run(w1)))

    print('\nOptimization Finished!')
    training_cost = sess.run(cost, feed_dict={X: X_train, y: y_train})
    print('Training cost = {:.9f} w0 = {:.6f} w1 = {:.6f}'.format(
        training_cost, sess.run(w0), sess.run(w1)))

    prediction = sess.run(y_pred, feed_dict={X: X_train})
    plt.plot(X_train, y_train, 'o', label='Training data')
    plt.plot(X_train, prediction, 'C2', label='Regression line')

    X_test = np.linspace(0, 50, 20) + np.random.normal(0, 3, 20)
    y_test = np.linspace(0, 50, 20) + np.random.normal(0, 3, 20)
    testing_cost = sess.run(cost, feed_dict={X: X_test, y: y_test})
    print('Testing cost =', testing_cost)
    print('Absolute difference =', abs(training_cost - testing_cost))

    plt.plot(X_test, y_test, '^', label='Testing data')
    plt.legend()