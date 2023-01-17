import tensorflow as tf
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
import pandas as pd

import numpy as  np


train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

X_train=train.drop('label',axis=1)
X_train=X_train.values.astype(np.float)*1.0/255.0

X_train
X_train.shape
train.label.reshape(-1,1)
def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

labels_flat = train[[0]].values.ravel()

X_label = dense_to_one_hot(labels_flat, 10)

X_label = X_label.astype(np.uint8)

X_label
sess=tf.InteractiveSession()
def weight_variable(shape):

    initial=tf.truncated_normal(shape,stddev=0.1)

    return tf.Variable(initial)
def bias_variable(shape):

    initial=tf.constant(0.1,shape=shape)

    return tf.Variable(initial)
def conv2d(x,W):

    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2X2(x):

    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
x=tf.placeholder(tf.float32,[None,784])

y_=tf.placeholder(tf.float32,[None,10])

X_image=tf.reshape(x,[-1,28,28,1])
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(X_image,W_conv1)+b_conv1)
h_pool1=max_pool_2X2(h_conv1)
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2X2(h_conv2)
W_fc1=weight_variable([7*7*64,1024])

b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])

h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
keep_prob=tf.placeholder(tf.float32)

h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
W_fc2=weight_variable([1024,10])

b_fc2=bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
tf.global_variables_initializer().run()
predict=tf.argmax(y_conv,1)
epochs_completed = 0

index_in_epoch = 0

num_examples = X_train.shape[0]



# serve data by batches

def next_batch(batch_size):

    

    global X_train

    global X_label

    global index_in_epoch

    global epochs_completed

    

    start = index_in_epoch

    index_in_epoch += batch_size

    

    # when all trainig data have been already used, it is reorder randomly    

    if index_in_epoch > num_examples:

        # finished epoch

        epochs_completed += 1

        # shuffle the data

        perm = np.arange(num_examples)

        np.random.shuffle(perm)

        X_train = X_train[perm]

        X_label = X_label[perm]

        # start next epoch

        start = 0

        index_in_epoch = batch_size

        assert batch_size <= num_examples

    end = index_in_epoch

    return X_train[start:end], X_label[start:end]
for i in range(2000):

    batch_xs, batch_ys = next_batch(50) 

    train_step.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})

X_test=test.values.astype(np.float)*1.0/255.0

predicted_lables = np.zeros(X_test.shape[0])

for i in range(0,X_test.shape[0]//50):

    predicted_lables[i*50 : (i+1)*50] = predict.eval(feed_dict={x: X_test[i*50 : (i+1)*50], 

                                                                                keep_prob: 1.0})
predicted_lables
df=pd.DataFrame({'ImageId':range(1,len(X_test)+1),'Label':predicted_lables})
df.to_csv('submission.csv',index=False)