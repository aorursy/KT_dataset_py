import tensorflow as tf
import keras 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
import random
image_size = 1024
input_size = 331
def img_transf(imgs):
    if len(imgs.shape) == 4:
        for i in range(imgs.shape[0]):
            for j in range(imgs.shape[-1]):
                imgs[i,...,j] /= imgs[i,...,j].max()
    elif len(imgs.shape) == 3 or 2:
        for j in range(imgs.shape[-1]):
            imgs[...,j] /= imgs[...,j].max()
    else:
        print('Input shape not recognised')
    return imgs
def rand_crop(img):
    h = random.randint(input_size, image_size) 
    cx = random.randint(0, image_size-h)
    cy = random.randint(0, image_size-h)
    cropped_img = img[cx:cx+h,cy:cy+h,:]
    return cv2.resize(cropped_img, (input_size,input_size))
from keras.preprocessing.image import ImageDataGenerator
data_dir = '../input/neuron cy5 full/Neuron Cy5 Full'

data_gen = ImageDataGenerator(horizontal_flip=True, #augmentation turned off for now
                              vertical_flip=True,
                              validation_split=0.2,
                              preprocessing_function = img_transf)

anchor_train_gen = data_gen.flow_from_directory(data_dir, 
                                                target_size=(input_size, input_size),
                                                color_mode='grayscale',
                                                class_mode='categorical',
                                                batch_size=16, 
                                                shuffle=True, 
                                                subset='training')
anchor_test_gen = data_gen.flow_from_directory(data_dir, 
                                               target_size=(input_size, input_size),
                                               color_mode='grayscale',
                                               class_mode='categorical',
                                               batch_size=16, 
                                               shuffle=True, 
                                               subset='validation')

train_gen = data_gen.flow_from_directory(data_dir, 
                                         target_size=(input_size, input_size),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=1, 
                                         shuffle=True, 
                                         subset='training')
test_gen = data_gen.flow_from_directory(data_dir, 
                                        target_size=(input_size, input_size),
                                        color_mode='grayscale',
                                        class_mode='categorical',
                                        batch_size=1, 
                                        shuffle=True, 
                                        subset='validation')

classes = dict((v, k) for k, v in anchor_train_gen.class_indices.items())
num_classes = len(classes)

def lossless_triplet_loss(y_true, y_pred, N = 2, beta=2, epsilon=1e-10):
    """
    Implementation of the triplet loss function
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    N  --  The number of dimension 
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)
    
    
    Returns:
    loss -- real number, value of the loss
    """
    anchor = tf.convert_to_tensor(y_pred[:,0:N])
    positive = tf.convert_to_tensor(y_pred[:,N:N*2]) 
    negative = tf.convert_to_tensor(y_pred[:,N*2:N*3])
    
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),1)
    
    #Non Linear Values  
    
    # -ln(-x/N+1)
    pos_dist = -tf.log(-tf.divide((pos_dist),beta)+1+epsilon)
    neg_dist = -tf.log(-tf.divide((N-neg_dist),beta)+1+epsilon)
    
    # compute loss
    loss = neg_dist + pos_dist
    
    return loss
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.layers import Input, Concatenate, Dense
from tensorflow.python.keras.optimizers import Adam

def create_model():
    vgg16_model = VGG16(include_top=False,
                         pooling='max',
                         input_shape=(input_size, input_size, 3),
                         weights='imagenet')
    outp = Dense(2, activation='sigmoid')(vgg16_model.output)
    pretrained_model = Model(vgg16_model.input, outp)
    cfg = pretrained_model.get_config()
    cfg['layers'][0]['config']['batch_input_shape'] = (None, input_size, input_size, 1)
    model = Model.from_config(cfg)
    for i, layer in enumerate(model.layers):
        if len(model.layers)-i < 7:
            layer.trainable = True
        else:
            layer.trainable = False
    return model

anc_inp = Input(shape=(input_size, input_size,1))
pos_inp = Input(shape=(input_size, input_size,1))
neg_inp = Input(shape=(input_size, input_size,1))

vgg16_model = create_model()
anc_outp = vgg16_model(anc_inp)
pos_outp = vgg16_model(pos_inp)
neg_outp = vgg16_model(neg_inp)

merged_outp = Concatenate(axis=-1)([anc_outp, pos_outp, neg_outp])

model = Model(inputs=[anc_inp, pos_inp, neg_inp], outputs=merged_outp)
adam_fine = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #10x smaller than standard
model.compile(optimizer='adam', loss=lossless_triplet_loss)
model.summary()
def triplet_gen(anchor_gen, gen):
    while True:
        anchors, y_anc = next(anchor_gen)
        pos = np.empty(anchors.shape)
        neg = np.empty(anchors.shape)
        for sample_idx in range(anchors.shape[0]):
            while pos[sample_idx].any() == None or neg[sample_idx].any() == None:
                img, y = next(gen)
                if y == y_anc[sample_idx]:
                    pos[sample_idx,...] = img
                else:
                    neg[sample_idx,...] = img
        yield [anchors, pos, neg], y_anc
def crop_gen(batches): #not used
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], input_size, input_size, 1))
        for i in range(batch_x.shape[0]):
            batch_crops[i,...,0] = rand_crop(batch_x[i])
        yield (batch_crops, batch_y)
history = model.fit_generator(triplet_gen(anchor_train_gen, train_gen), #inefficient, will change 
                              epochs=5,
                              steps_per_epoch=4*len(anchor_train_gen),
                              validation_data=triplet_gen(anchor_test_gen, test_gen),
                              validation_steps=len(anchor_test_gen), 
                              verbose=1)
X ,y = next(triplet_gen(anchor_test_gen, test_gen))
points = model.predict(X, verbose=1)[:,0:2]
plt.scatter(points[:,0], points[:,1], c=y[:,0])
for layer in model.layers:
    layer.trainable = True
model.compile(optimizer=adam_fine, loss=lossless_triplet_loss)
history2 = model.fit_generator(triplet_gen(anchor_train_gen, train_gen), #inefficient, will change 
                              epochs=10,
                              steps_per_epoch=4*len(anchor_train_gen),
                              validation_data=triplet_gen(anchor_test_gen, test_gen),
                              validation_steps=len(anchor_test_gen), 
                              verbose=1)
X ,y = next(triplet_gen(anchor_test_gen, test_gen))
points = model.predict(X, verbose=1)[:,0:2]
plt.scatter(points[:,0], points[:,1], c=y[:,0])