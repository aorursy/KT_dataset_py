import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import tensorflow as tf
import math

import cv2

# Some constants 
INPUT_FOLDER = '../input/sample_dataset_for_testing/sample_dataset_for_testing/fullsampledata/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
IMAGE_PX_SIZE = 100
patients
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, int(len(l)), int(n)):
        yield l[int(i):int(i + n)]


def mean(a):
    return sum(a) / len(a)
def load_slices(path):
    if_mask_label = False
    slices = []
    slices = [np.sum(cv2.resize(mpimg.imread(path + '/' +images) , (50,50)) ,axis=2) for images in sorted(os.listdir(path))]
#     slices = [cv2.resize(np.sum(mpimg.imread(path + '/' +images) ,axis=2),(IMAGE_PX_SIZE,IMAGE_PX_SIZE)) for images in sorted(os.listdir(path))]
    for i in sorted(os.listdir(path)):
        if 'mask' in re.split('_|\.',i):
            if_mask_label = True
            break
    return slices,if_mask_label
    
all_patients_path = []
#Listing patients
for i in range(0,len(patients)):
    path = INPUT_FOLDER + patients[i]
    each_patient = os.listdir(path)
    all_patients_path.append(INPUT_FOLDER + patients[i] + '/' + each_patient[0])
    print('In ',patients[i])
    print('Numer of patients ' , len(each_patient))
    print('TIFF files for ' , each_patient[0]  , len(os.listdir(path + '/' + each_patient[0])))
    print('\n')
    slices = []
    #print(each_patient)
complete_data = []
no_of_slices = 10
for patients in all_patients_path:
    scan_slices, is_abnormality = load_slices(patients)
    if len(scan_slices)< 290:
        for i in range(len(scan_slices),290):
            scan_slices.append(scan_slices[i-1])
    else:
        last_element = sum(scan_slices[289:])/len(scan_slices[289:]) 
        scan_slices = np.delete(scan_slices,np.s_[289:len(scan_slices)-1],axis=0)
        np.append(scan_slices,last_element)
    
    new_slices = []
    
    chunk_sizes = math.ceil(len(scan_slices) / no_of_slices)
    for slice_chunk in chunks(scan_slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)
        
    print('Shape ' , np.array(new_slices).shape)

    complete_data.append([np.array(new_slices),int(is_abnormality)])    
    #print(str(len(scan_slices)) +" "+ str(is_abnormality))
# NN
IMG_SIZE_PX = 50
SLICE_COUNT = 10
n_classes = 2
batch_size = 1

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([25000,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)


    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 25000])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output
train_data = complete_data[:8]
validation_data = complete_data[8:]


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    
    hm_epochs =5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        successful_runs = 0
        total_runs = 0
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one 
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    pass
                    #print(str(e))
            
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

#             print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
            
        print('Done. Finishing accuracy:')
#         print('Accuracy:',accuracy.eval({x:[validation_data[0][0]], y:[validation_data[0][1]]}))
        

# Run this locally:
train_neural_network(x)
for i in validation_data:
    print(i[1])
validation_data[0][1]
