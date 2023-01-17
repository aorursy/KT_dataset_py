import pandas as pd
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet


def extract_images_and_labels(dataset, validation = False):
    
    images = dataset[:, 1:].reshape(-1, 28, 28, 1)

    labels_dense = dataset[:, 0]
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * 10
    labels_one_hot = np.zeros((num_labels, 10))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    if validation:
        num_images = images.shape[0]
        divider = num_images - 200
        return images[:divider], labels_one_hot[:divider], images[divider+1:], labels_one_hot[divider+1:]
    else:
        return images, labels_one_hot

def extract_images(dataset):
    return dataset.reshape(-1, 28*28)

train_data_file = '../input/train.csv'
test_data_file = '../input/test.csv'

train_data = pd.read_csv(train_data_file).as_matrix().astype(np.uint8)
test_data = pd.read_csv(test_data_file).as_matrix().astype(np.uint8)

train_images, train_labels, val_images, val_labels = extract_images_and_labels(train_data, validation=True)
test_images = extract_images(test_data)

#创建DataSet对象
train = DataSet(train_images,
                train_labels,
                dtype = np.float32,
                reshape = True)

validation = DataSet(val_images,
                val_labels,
                dtype = np.float32,
                reshape = True)

test = test_images
print("Finish!")
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

def get_batchs(data, batch_size):
    size = data.shape[0]
    for i in range(size//batch_size):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:]
        else:
            yield data[i*batch_size:(i+1)*batch_size]




x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(2000):
  batch = train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: validation.images, y_: validation.labels, keep_prob: 1.0}))
    
# 输出预测的测试集标签

f = open('/kaggle/working/prediction.csv', 'w+')
f.write('ImageId,Label\n')
batchs = get_batchs(test, 50)
i = 1
for test_image in batchs:
    prediction = tf.argmax(y_conv, 1)
    test_labels = prediction.eval(feed_dict={x: test_image, keep_prob: 1.0})
    for label in test_labels:
        f.write(str(i) + ',' +str(label) + '\n')
        i += 1
f.close()
