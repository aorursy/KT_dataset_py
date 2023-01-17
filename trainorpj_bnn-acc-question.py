# computational

import numpy as np

import tensorflow as tf

import tensorflow_probability as tfp



# plotting

import matplotlib.pyplot as plt

%matplotlib inline



# datasets

from sklearn.datasets import make_moons

from sklearn.model_selection import train_test_split



# aliased for ease-of-use

tfk = tf.keras

tfd = tfp.distributions
# moons dataset: (x1, x2), (label)

xs, labels = make_moons(n_samples=2000, noise=0.3, random_state=27)

xs = xs.astype(np.float32)

labels = labels.reshape(-1, 1).astype(np.int32)



x_train, x_test, y_train, y_test = train_test_split(xs, labels, 

                                                    test_size=0.2, random_state=27)



plt.figure(figsize=(10, 6))

plt.scatter(xs[:, 0], xs[:, 1], 

            color=["red" if lb==0 else "blue" for lb in labels[:, 0]], 

            alpha=0.8)
tf.reset_default_graph()



# x1, x2

n_features = 2



bnn = tfk.Sequential([

    tfp.layers.DenseFlipout(5, 

                            input_shape=(n_features,),

                            activation=tf.nn.tanh),

    tfp.layers.DenseFlipout(5, 

                            activation=tf.nn.tanh),

    tfp.layers.DenseFlipout(1)

])



bnn.summary()
# x and y placeholders

x = tf.placeholder(shape=[None, 2], dtype=tf.float32)

y = tf.placeholder(shape=[None, 1], dtype=tf.int32)



# predict logits

logits = bnn(x)

# set as parameters of Bernoulli dist

labels_distribution = tfd.Bernoulli(logits=logits)



# calculate ELBO loss

neg_log_lik = -tf.reduce_mean(labels_distribution.log_prob(y))

kl = tf.reduce_mean(bnn.losses)

elbo_loss = neg_log_lik + kl



# minimize ELBO loss

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

train_op = optimizer.minimize(elbo_loss)



# determine predictions

predictions = tf.cast(logits > 0, dtype=tf.int32)

correct_predictions = tf.equal(predictions, y)

accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
n_iter = 800

show_step = 100



init = tf.global_variables_initializer()



with tf.Session() as sess:

    sess.run(init)

    

    # record losses and accuracy

    history_loss_train = []

    history_loss_test = []

    history_acc_train = []

    history_acc_test = []

            

    w_history = []



    print("Training")

    for step in range(n_iter):

        feed_dict = {x: x_train, y: y_train}

        

        sess.run(train_op, feed_dict=feed_dict)

        

        loss_train = sess.run(elbo_loss, feed_dict=feed_dict)

        acc_train = sess.run(accuracy, feed_dict=feed_dict)

        

        history_loss_train.append(loss_train)

        history_acc_train.append(acc_train)

        

        if (step + 1) % show_step == 0:

            print("-" * 50)

            print ('Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}'.format(step+1, loss_train, acc_train))

        

        feed_dict = {x: x_test, y: y_test}

        

        loss_test = sess.run(elbo_loss, feed_dict=feed_dict)

        acc_test = sess.run(accuracy, feed_dict=feed_dict)

        

        history_loss_test.append(loss_test)

        history_acc_test.append(acc_test)

        

        # end training loop
fig = plt.figure(figsize = (15, 6))

ax1 = fig.add_subplot(1, 2, 1)

ax1.plot(range(n_iter), history_loss_train, label="Training")

ax1.plot(range(n_iter), history_loss_test, label="Test")

ax1.set_title("Loss")

ax1.legend(loc = "upper right")



ax2 = fig.add_subplot(1, 2, 2)

ax2.plot(range(n_iter), history_acc_train, label = "Training")

ax2.plot(range(n_iter), history_acc_test, label = "Test")

ax2.set_ylim(0.0, 1.0)

ax2.set_title("Accuracy")

ax2.legend(loc = "lower right")



plt.show()