#import libraries
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from matplotlib import pyplot as plt
import numpy as np
import time
import os
import cv2
import random
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from keras_preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from sklearn.utils import shuffle
#define the parameters
img_size = 160
keep_prob = 0.5
learning_rate = 0.00001
epochs = 150
test_epoch = 10
batch_size = 256
x_input = tf.placeholder(tf.float32, [None, 160, 160, 3], name="x_input")
y_input = tf.placeholder(tf.int32, [None], name="y_input")

#define the data path
train_path = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train/"
test_path = "../input/chest-xray-pneumonia/chest_xray/chest_xray/test/"
val_path = "../input/chest-xray-pneumonia/chest_xray/chest_xray/val/"
save_path = "../output/kaggle/working/"
#data link : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
#define the read data function
def read_data(file_path):
    data_x = []
    data_y = []
    for folder_name in os.listdir(file_path):
        #print(folder_name)
        if folder_name.startswith('.'):
            continue
        if folder_name == 'NORMAL':
            label = 0
        elif folder_name == 'PNEUMONIA':
            label = 1
        path = file_path+folder_name
        img_list = os.listdir(path)
        for img_name in img_list:
            img = cv2.imread(path+"/"+img_name)
            if img is None : continue
            img = cv2.resize(img, (img_size,img_size))
            data_x.append(np.asarray(img))
            data_y.append(label)
    return np.asarray(data_x), np.asarray(data_y)

# define the function to convert the number to one-hot code
def convert_onehot(arr, num):
    return (np.arange(num)==arr[:,None]).astype(np.integer)

#read the data from the file
x_train, y_train = read_data(train_path)
x_test, y_test = read_data(test_path)
x_val, y_val = read_data(val_path)

#shuffle the data
x_train, y_train = shuffle(x_train, y_train, random_state=0)
x_test, y_test = shuffle(x_test, y_test, random_state=0)
x_val, y_val = shuffle(x_val, y_val, random_state=0)

#normalize the data
x_train = x_train/255
x_test = x_test/255
x_val = x_val/255

#we observe the class in training data
print("Pneumonia train: " + str(np.count_nonzero(y_train == 1)))
print("Normal train: " + str(np.count_nonzero(y_train == 0)))
#because the class in training data is imbalanced, so we use oversampling.
tmpShape = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
x_trainFlat = x_train.reshape(x_train.shape[0], tmpShape)
ros = RandomOverSampler(random_state=1)
x_trainRos, y_trainRos = ros.fit_resample(x_trainFlat, y_train)
x_trainData = x_trainRos.reshape(x_trainRos.shape[0], img_size, img_size, 3)
y_trainData = y_trainRos

#observe the class in training data after oversampling
print("Pneumonia train: " + str(np.count_nonzero(y_trainData == 1)))
print("Normal train: " + str(np.count_nonzero(y_trainData == 0)))
#define a convenient function for initial the conv weights
def conv_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1),name)

#define a convenient function for initial the conv bias
def conv_bias(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name)

#define a convenient function for the conv operation
def conv_op(x, W, padding, name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding, name=name)

#define a convenient function for the relu activation operation
def conv_relu(x, bias, name):
    return tf.nn.relu(tf.nn.bias_add(x, bias), name=name)

#define a convenient function for the pooling operation
def max_pooling(x,  n, name):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding="SAME", name=name)
#define the training network
def train_network(x):
    # define the 1st layer(conv)
    x = tf.reshape(x, [-1, 160, 160, 3])
    with tf.name_scope("L1_conv"):
        # input is 160x160x3, the kernel is 3x3x32 with padding, so the output is 160x160x32
        layer1_weights = conv_weights([3, 3, 3, 32], "layer1_conv_weights")
        layer1_bias = conv_bias([32], "layer1_conv_bias")
        layer1_conv_res = conv_op(x, layer1_weights, "SAME", "layer1_conv_op")
        layer1_conv_relu = conv_relu(layer1_conv_res, layer1_bias, "layer1_conv_relu")

    with tf.name_scope("L2_pooling"):
        # the input is 160x160x32, the pool is 2x2 with stride =2, so the output is 80x80x32
        layer2_pool = max_pooling(layer1_conv_relu, 2, "layer2_pool")

    with tf.name_scope("L3_conv"):
        # input is 80x80x32, the kernel is 3x3x64 with padding, so the output is 80x80x64
        layer3_weights = conv_weights([3, 3, 32, 64], "layer3_conv_weights")
        layer3_bias = conv_bias([64], "layer3_conv_bias")
        layer3_conv_res = conv_op(layer2_pool, layer3_weights, "SAME", "layer3_conv_op")
        layer3_conv_relu = conv_relu(layer3_conv_res, layer3_bias, "layer3_conv_relu")

    with tf.name_scope("L4_pooling"):
        # input is 80x80x64, the pool is 2x2 with stride = 2, so the output is 40x40x64
        layer4_pool = max_pooling(layer3_conv_relu, 2, "layer4_pool")

    with tf.name_scope("L5_conv"):
        # input is 40x40x64, the kernel is 3x3x128 with padding, so the output is 40x40x128
        layer5_weights = conv_weights([3, 3, 64, 128], "layer5_conv_weights")
        layer5_bias = conv_bias([128], "layer5_conv_bias")
        layer5_conv_res = conv_op(layer4_pool, layer5_weights, "SAME", "layer5_conv_op")
        layer5_conv_relu = conv_relu(layer5_conv_res, layer5_bias, "layer5_conv_relu")

    with tf.name_scope("L6_pooling"):
        # input is 40x40x128, the pool is 2x2 with stride = 2, so the output is 20x20x128
        layer6_pool = max_pooling(layer5_conv_relu, 2, "layer6_pool")

    with tf.name_scope("L7_conv"):
        # input is 20x20x128, the kernel is 3x3x256 with padding, the output is 20x20x256
        layer7_weights = conv_weights([3, 3, 128, 256], "layer7_conv_weights")
        layer7_bias = conv_bias([256], "layer7_conv_bias")
        layer7_conv_res = conv_op(layer6_pool, layer7_weights, "VALID", "layer7_conv_op")
        layer7_conv_relu = conv_relu(layer7_conv_res, layer7_bias, "layer7_conv_relu")

    with tf.name_scope("L8_pooling"):
        # input is 20x20x256, the pool is 2x2 with stride = 2, so the output is 10x10x256
        layer8_pool = max_pooling(layer7_conv_relu, 2, "layer8_pool")

    with tf.name_scope("L9_conv"):
        # input is 10x10x256, the kernel is 3x3x512 with padding, so the output is 10x10x512
        layer9_weights = conv_weights([3, 3, 256, 512], "layer9_conv_weights")
        layer9_bias = conv_bias([512], "layer9_conv_bias")
        layer9_conv_res = conv_op(layer8_pool, layer9_weights, "SAME", "layer9_conv_op")
        layer9_conv_relu = conv_relu(layer9_conv_res, layer9_bias, "layer9_conv_relu")

    with tf.name_scope("L10_pool"):
        # input is 10x10x512, the pool is 2x2 with stride = 2, so the ouput is 5x5x512
        layer10_pool = max_pooling(layer9_conv_relu, 2, "layer10_pool")

    with tf.name_scope("L11_conv"):
        # input is 5x5x512, the kernel is 5x5x40996, and reshape it to 1x4096
        layer11_weights = conv_weights([5, 5, 512, 4096], "layer11_conv_weights")
        layer11_bias = conv_bias([4096], "layer11_conv_bias")
        layer11_conv_res = conv_op(layer10_pool, layer11_weights, "VALID", "layer11_conv_op")
        layer11_conv_relu = conv_relu(layer11_conv_res, layer11_bias, "layer11_conv_relu")
        layer11_conv_flat= tf.reshape(layer11_conv_relu, [-1, 4096])

    with tf.name_scope("L12_fc"):
        # input is 1x4096, output is 1x128
        layer12_fc = tf.layers.dense(layer11_conv_flat, 128, activation=tf.nn.relu, trainable=True, use_bias=True)
        layer12_drop = tf.nn.dropout(layer12_fc, keep_prob=keep_prob)
        #print(layer12_fc)
    with tf.name_scope("L13_fc"):
        layer13_fc = tf.layers.dense(layer12_drop, 2, activation=tf.nn.sigmoid, trainable=True, use_bias=True)
    return layer13_fc
pred = train_network(x_input)

#compute the loss and cost
with tf.name_scope("loss"):
    #cost = tf.losses.mean_squared_error(y_input, pred)
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y_input)
    loss = tf.reduce_mean(cost)

#define the data augmentaion function
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

#the number of training images is 7756, which is not enough, so we use data augmentation to produce more training data.
data_flow = datagen.flow(x_trainData, y_trainData, batch_size=batch_size)

#do the gradient descent
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gradients = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(gradients, global_step=global_step)

#compute the accuracy
correct_prediction = tf.nn.in_top_k(pred, y_input, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#start training process
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_lo = []
    train_ac = []
    test_lo = []
    test_ac = []
    val_lo = []
    val_ac = []
    train_index = []
    test_index =[]
    for i , (x_tr, y_tr) in enumerate(data_flow):
        if i == epochs:
            break
        #do training
        _, pre, train_loss, train_accuracy = sess.run([train_op, pred, loss, accuracy],
                                        feed_dict={x_input: x_tr, y_input: y_tr})
        train_lo.append(train_loss)
        train_ac.append(train_accuracy)
        train_index.append(i)
        print("epoch %d train loss :%g, train accuracy: %g" %(i, train_loss, train_accuracy))
        #we define a test/eval epoch time.
        if i % test_epoch == 0:
            test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict={x_input: x_test, y_input: y_test})
            print("epoch %d test loss : %g, test accuracy : %g" % (i, test_loss, test_accuracy))
            val_loss, val_accuracy = sess.run([loss, accuracy], feed_dict={x_input: x_val, y_input: y_val})
            print("epoch %d eval loss : %g, eval accuracy : %g" %(i, val_loss, val_accuracy))
            test_lo.append(test_loss)
            test_ac.append(test_accuracy)
            val_lo.append(val_loss)
            val_ac.append(val_accuracy)
            test_index.append(i)
#plot the loss
plt.plot(train_index, train_lo, color='r', label="training loss")
plt.plot(test_index, test_lo, color='b', label="testing loss")
plt.plot(test_index, val_lo, color='y', label="val loss")
plt.title('model loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
#plot the accuracy
plt.plot(train_index, train_ac, color='r', label="training accuracy")
plt.plot(test_index, test_ac, color='b', label="testing accuracy")
plt.plot(test_index, val_ac, color='y', label="val accuracy")
plt.title('model accuracy')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()