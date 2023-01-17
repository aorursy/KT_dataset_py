from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import math

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



%matplotlib inline
grasp_df = pd.read_csv('../input/shadow_robot_dataset.csv')

grasp_df.describe()
exp_mea_grasp_df = grasp_df[['experiment_number', ' measurement_number']]

grouped = exp_mea_grasp_df.groupby('experiment_number')



print('Number of experiments:', grouped.agg(np.max).values.shape[0])

print('Max measurements:', max(grouped.agg(np.max).values)[0] + 1)

print('Aprox. Mean measurements:', np.mean(grouped.agg(np.max).values))

print('Aprox. Std measurements:', np.std(grouped.agg(np.max).values))
robustness_values = grasp_df.values[:, 1];

robustness_values = robustness_values.reshape((robustness_values.shape[0]), 1)

robustness_values.shape



print('Min robustness:', min(robustness_values))

print('Max robustness:', max(robustness_values))



plt.hist(robustness_values)

plt.title('Robustness Histogram')

plt.xlabel('Robustness')

plt.ylabel('Frequency')
categories = 4

min_value = min(robustness_values)

max_value = max(robustness_values)

range_values = max_value - min_value

range_step = range_values / categories

q1 = min_value + range_step

q2 = min_value + 2 * range_step

q3 = min_value + 3 * range_step

q4 = min_value + 4 * range_step



new_robustness_values = []



for i in range(robustness_values.shape[0]):

    value = robustness_values[i]

    

    if value <= q1:

        new_robustness_values.append(0)

    elif value <= q2:

        new_robustness_values.append(1)

    elif value <= q3:

        new_robustness_values.append(2)

    elif value <= q4:

        new_robustness_values.append(3)

        

print('Category 0:', new_robustness_values.count(0))

print('Category 1:', new_robustness_values.count(1))

print('Category 2:', new_robustness_values.count(2))

print('Category 3:', new_robustness_values.count(3))
robustness_values_c0 = robustness_values[robustness_values <= q1] # category 0 condition

robustness_values_c3 = robustness_values[robustness_values > q3] # category 3 condition



plt.figure(figsize = (15, 5))



plt.subplot(121)

plt.hist(robustness_values_c0)

plt.title('Robustness 0 Histogram')

plt.xlabel('Robustness')

plt.ylabel('Frequency')



plt.subplot(122)

plt.hist(robustness_values_c3)

plt.title('Robustness 3 Histogram')

plt.xlabel('Robustness')

plt.ylabel('Frequency')
q1s = np.ones(grasp_df[' robustness'].shape) * q1

clean_grasp_df = grasp_df[grasp_df[' robustness'] <= q1s]

clean_grasp_df.describe()
clean_grasp_df.drop('experiment_number', axis = 1, inplace = True)

clean_grasp_df.drop(' measurement_number', axis = 1, inplace = True)



label_data = np.copy(clean_grasp_df[' robustness'].values)

label_data[label_data < 100] = 0;

label_data[label_data >= 100] = 1;



plt.hist(label_data)

plt.title('Categories distribution')

plt.xlabel('Robustness (Bad/Good)')

plt.ylabel('Frequency')

plt.xticks([0, 1])
grasp_data = (clean_grasp_df - clean_grasp_df.mean()) / clean_grasp_df.std()

grasp_data.describe()
# Just get from grasp_data the feature columns as a NumPy array

x_train, x_test, y_train, y_test = train_test_split(grasp_data.values[:, 1:], label_data, 

                                                    train_size = 0.80, test_size = 0.20,

                                                    random_state = 0)



print('Train shape', x_train.shape)

print('Test shape', x_test.shape)
samples = x_train.shape[0]

input_dim = x_train.shape[1]



# hyper-parameters

learning_rate = 0.001

max_epochs = 40

batch_size = 512

labels = np.unique(y_train).shape[0]

batch_epsilon = 1e-3



# Inputs

input_x = tf.placeholder(tf.float32, [None, input_dim])

input_y = tf.placeholder(tf.int32, [None])



# 1st layer

hidden_units1 = 1024



w1 = tf.Variable(tf.truncated_normal([input_dim, hidden_units1],

                                     stddev = math.sqrt(2.0 / input_dim)))

z1 = tf.matmul(input_x, w1)



batch_mean1, batch_var1 = tf.nn.moments(z1, [0])

beta1 = tf.Variable(tf.zeros([hidden_units1]))

scale1 = tf.Variable(tf.ones([hidden_units1]))

batch_norm1 = tf.nn.batch_normalization(z1, batch_mean1, batch_var1, beta1, scale1, batch_epsilon)



h1 = tf.nn.relu(batch_norm1)



# 2nd layer

hidden_units2 = 1024



w2 = tf.Variable(tf.truncated_normal([hidden_units1, hidden_units2],

                                     stddev = math.sqrt(2.0 / hidden_units1)))

z2 = tf.matmul(h1, w2)



batch_mean2, batch_var2 = tf.nn.moments(z2, [0])

beta2 = tf.Variable(tf.zeros([hidden_units2]))

scale2 = tf.Variable(tf.ones([hidden_units2]))

batch_norm2 = tf.nn.batch_normalization(z2, batch_mean2, batch_var2, beta2, scale2, batch_epsilon)



h2 = tf.nn.relu(batch_norm2)



# 3rd layer

hidden_units3 = 512



w3 = tf.Variable(tf.truncated_normal([hidden_units2, hidden_units3],

                                     stddev = math.sqrt(2.0 / hidden_units2)))

z3 = tf.matmul(h2, w3)



batch_mean3, batch_var3 = tf.nn.moments(z3, [0])

beta3 = tf.Variable(tf.zeros([hidden_units3]))

scale3 = tf.Variable(tf.ones([hidden_units3]))

batch_norm3 = tf.nn.batch_normalization(z3, batch_mean3, batch_var3, beta3, scale3, batch_epsilon)



h3 = tf.nn.relu(batch_norm3)



# Last layer

wOut = tf.Variable(tf.truncated_normal([hidden_units3, labels],

                                       stddev = math.sqrt(2.0 / hidden_units3)))

bOut = tf.Variable(tf.zeros([labels]))

logits = tf.nn.bias_add(tf.matmul(h3, wOut), bOut)



# Loss and optimiser

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,

                                                                        labels = input_y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

train_op = optimizer.minimize(loss_op)



# Accuracy

prediction = tf.cast(tf.argmax(logits, 1), tf.int32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, input_y), tf.float32))
losses = []

train_accuracies = []



# Initializing the variables

init = tf.global_variables_initializer()



config = tf.ConfigProto()

config.gpu_options.allow_growth = True



with tf.Session(config = config) as sess:

    sess.run(init)



    # Training cycle

    for epoch in range(max_epochs):        

        avg_loss = 0.0

        total_batch = int(samples / batch_size)

        

        # Loop over all batches

        for i in range(total_batch):

            batch_index = np.random.choice(samples, batch_size, replace = False)

            batch_x = x_train[batch_index]

            batch_y = y_train[batch_index].flatten()



            _, loss, acc = sess.run([train_op, loss_op, accuracy],

                                    feed_dict = {input_x: batch_x, input_y: batch_y})

            

            avg_loss += loss / total_batch

            

        print("Epoch={:02d}".format(epoch+1), "loss={:.9f}".format(avg_loss), 

              "train acc={:.8f}".format(acc))

        

        losses.append(avg_loss)

        train_accuracies.append(acc)

        

    # Evaluate on test set

    # TODO: THIS IS NOT BEING DONE TOTALLY RIGHT, BATCH NORM MEAN/VAR SHOULD BE TAKEN

    # FROM AN EXPONENTIALLY WEIGHTED AVERAGE ACCROSS MINI BATCHES DURING TRAINING SO NOW WE

    # MUST RE BUILD THE GRAPH BUT SUBSTITUTING THESE VALUES WITH THE AVERAGED ONES

    # BEFORE EVALUATING ON THE TEST SET. HOWEVER, SINCE WE HAVE +100k TEST SAMPLES,

    # THEIR MEAN/VAR WILL BE STILL A GOOD APPROXIMATION TO THE WHOLE POPULATION VALUES.

    

    batch_x = x_test

    batch_y = y_test.flatten()

    acc_test = sess.run([accuracy], feed_dict = {input_x: batch_x, input_y: batch_y})[0]

    

    print("Accuracy on test set:", acc_test)
plt.figure(figsize = (15, 5))



plt.subplot(121)

plt.plot(losses)

plt.title('Train loss evolution')

plt.xlabel('Epoch')

plt.ylabel('Loss')



plt.subplot(122)

plt.plot(train_accuracies)

plt.title('Train accuracy evolution')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')