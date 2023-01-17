!pip install tensorflow==1.14
# computational

import numpy as np

import tensorflow as tf

import tensorflow_probability as tfp

from sklearn.preprocessing import StandardScaler



# functional programming

from functools import partial



# plotting

import matplotlib.pyplot as plt



# plot utilities

%matplotlib inline

plt.style.use('ggplot')

def plt_left_title(title): plt.title(title, loc="left", fontsize=18)

def plt_right_title(title): plt.title(title, loc='right', fontsize=13, color='grey')



# use eager execution for better ease-of-use and readability

# tf.enable_eager_execution()



# aliases

tfk = tf.keras

tfd = tfp.distributions



# helper function, mostly for plotting

def logistic(x, w, b):

    exp_term = np.exp(-(x * w + b))

    return 1 / (1 + exp_term)



print(f"            tensorflow version: {tf.__version__}")

print(f"tensorflow probability version: {tfp.__version__}")
data = np.array([[66.,  0.],

 [70.,  1.],

 [69.,  0.],

 [68.,  0.],

 [67.,  0.],

 [72.,  0.],

 [73.,  0.],

 [70.,  0.],

 [57.,  1.],

 [63.,  1.],

 [70.,  1.],

 [78.,  0.],

 [67.,  0.],

 [53.,  1.],

 [67.,  0.],

 [75.,  0.],

 [70.,  0.],

 [81.,  0.],

 [76.,  0.],

 [79.,  0.],

 [75.,  1.],

 [76.,  0.],

 [58.,  1.]])



# xs: temperature

# ys: o-ring failure (1 == failure occurred)

n_observations = data.shape[0]

xs, ys = data[:, 0, np.newaxis], data[:, 1, np.newaxis]

scaler = StandardScaler()

xs = scaler.fit_transform(xs)

xs_test = np.linspace(xs.min(), xs.max(), 53)[:, np.newaxis]



def plot_training_data(): 

    plt.figure(figsize=(12, 7))

    plt.scatter(xs, ys, c="#619CFF", label="observed", s=200)

    plt.xlabel("Temperature (scaled)")

    plt.ylabel("O-Ring Failure")

    

def plt_left_title(title): plt.title(title, loc="left", fontsize=18)

def plt_right_title(title): plt.title(title, loc='right', fontsize=13, color='grey')

    

print(f"{n_observations} observations")

print(f"{int(ys.sum())} failures")
# train an artificial neural net

nn_model = tfk.Sequential([

    tfk.layers.Dense(1,

                     activation=tf.nn.sigmoid)

])

nn_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01),

                 loss="binary_crossentropy")

nn_model.fit(xs, ys,

             epochs=800,

             verbose=False);

print(f"estimated w: {nn_model.layers[0].get_weights()[0]}")

print(f"estimated b: {nn_model.layers[0].get_weights()[1]}")
plot_training_data()

plt_left_title("Challenger Dataset")

plt_right_title("How can we quantify uncertainty in our prediction?")



# plot neural network result

plt.plot(xs_test,

         nn_model.predict(xs_test),

         "g", linewidth=4,

         label="artificial neural net prediction")

plt.legend(loc="center right");
# placeholder variables

x = tf.placeholder(shape=[None, 1], dtype=tf.float32)

y = tf.placeholder(shape=[None, 1], dtype=tf.int32)



# flipout layer, which will yield distributions on our weights

# in this case, only one weight and a bias term, each with normal priors

layer = tfp.layers.DenseFlipout(1, 

                                activation=None,

                                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),

                                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn())



# make a prediction

logits = layer(x)

# those predictions are parameters for bernoulli distributions 

labels_dist = tfd.Bernoulli(logits=logits)



# use evidence-lower bound (ELBO) as the loss

neg_log_likelihood = -tf.reduce_mean(labels_dist.log_prob(y))

kl = sum(layer.losses) / n_observations

elbo_loss = neg_log_likelihood + kl



# make predictions, and check accuracy

predictions = tf.cast(logits > 0, dtype=tf.int32)

correct_predictions = tf.equal(predictions, y)

accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))



# minimize ELBO

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

train_op = optimizer.minimize(elbo_loss)
n_steps = 1300

n_posterior_samples = 125



history_loss = []

history_acc = []



candidate_ws = []

candidate_bs = []



init_op = tf.global_variables_initializer()



with tf.Session() as sess:

    sess.run(init_op)

    

    # run training loop

    print("Start training...")

    for step in range(n_steps):

        # feed in training data

        feed_dict = {x: xs, y: ys}

        

        # execute the graph 

        _ = sess.run(train_op, feed_dict=feed_dict)

        

        # determine loss and accuracy

        loss_value, acc_value = sess.run([elbo_loss, accuracy],feed_dict=feed_dict)

        

        if ((step + 1) % 100) == 0:

            print(f"{'-'*50} step {step + 1}")

            print(f"Loss {loss_value:.3f}, Accuracy: {acc_value:.3f}")

        

        # record loss and accuracy

        history_loss.append(loss_value)

        history_acc.append(acc_value)

    

    print("Done training!\n")

    print(f"Taking {n_posterior_samples} samples from posterior distributions on weights\n")

    

    w_draw = layer.kernel_posterior.sample(seed=27)

    b_draw = layer.bias_posterior.sample(seed=27)

    

    for mc in range(n_posterior_samples):

        w_, b_ = sess.run([w_draw, b_draw])

        candidate_ws.append(w_)

        candidate_bs.append(b_)

        

    print("Sampling complete. Samples are stored in numpy arrays:")

    print(f"  weight: candidate_ws")

    print(f"    bias: candidate_bs")

        

candidate_ws = np.array(candidate_ws).reshape(-1, 1).astype(np.float32)

candidate_bs = np.array(candidate_bs).astype(np.float32)
fig = plt.figure(figsize=(14, 5))

ax1, ax2 = fig.subplots(1, 2)

ax1.plot(history_loss)

ax1.set_title("ELBO loss", loc="left", fontsize=18)



ax2.plot(history_acc)

ax2.set_title("Accuracy", loc="left", fontsize=18)

ax2.set_title("noise is expected", loc='right', fontsize=13, color='grey')



plt.show();
fig = plt.figure(figsize=(14, 5))

ax1, ax2 = fig.subplots(1, 2)



ax1.hist(candidate_ws);

ax1.set_title(f"$w$ posterior", loc="left", fontsize=18)

ax1.set_title(f"mean = {candidate_ws.mean():.3f}", loc='right', fontsize=13, color='grey')



ax2.hist(candidate_bs)

ax2.set_title(f"$b$ posterior", loc="left", fontsize=18)

ax2.set_title(f"mean = {candidate_bs.mean():.2f}", loc='right', fontsize=13, color='grey')



print(f"{n_posterior_samples} posterior samples")
plot_training_data()

plt_left_title("Challenger Data")

plt_right_title(f"{n_posterior_samples} draws from posterior")



# plot candidate curves

plt.plot(xs_test, 

         logistic(xs_test, candidate_ws.T, candidate_bs.T), 

         'r', alpha=0.2, linewidth=0.5);

# this is a placeholder for the labels -- no data is plotted

plt.plot([], [], 'r',

         label=f"candidate curves")

# plot candidate curve based on mean weights

plt.plot(xs_test, 

         logistic(xs_test, candidate_ws.mean(), candidate_bs.mean()), 

         '--', c='darkred', linewidth=4, 

         label="mean candidate curve")

# plot neural network result

plt.plot(xs_test,

         nn_model.predict(xs_test),

         "g", linewidth=4,

         label="artificial neural net")

plt.legend(loc="center right", facecolor="white");