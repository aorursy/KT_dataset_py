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

data_gen = ImageDataGenerator(horizontal_flip=True,
                              vertical_flip=True,
                              validation_split=0.1,
                              preprocessing_function = img_transf)
train_gen = data_gen.flow_from_directory(data_dir, 
                                         target_size=(image_size,image_size),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=32, 
                                         subset='training',
                                         shuffle=True)
test_gen = data_gen.flow_from_directory(data_dir, 
                                        target_size=(image_size, image_size),
                                        color_mode='grayscale',
                                        class_mode='categorical',
                                        batch_size=32, 
                                        subset='validation',
                                        shuffle=True)

classes = dict((v, k) for k, v in train_gen.class_indices.items())
num_classes = len(classes)
def crop_gen(batches):
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], input_size, input_size, 1))
        for i in range(batch_x.shape[0]):
            batch_crops[i,...,0] = rand_crop(batch_x[i])
        yield (batch_crops, batch_y)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.python.keras.optimizers import Adam

pretrained_model = ResNet50(include_top=False,
                         pooling='none',
                         input_shape=(input_size, input_size, 3),
                         weights='imagenet')
x = Flatten()(pretrained_model.output)
#x = Dense(4096, activation='relu')(x)
#x = Dropout(0.5)(x)
#x = Dense(2048, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)
resnet_model = Model(pretrained_model.input, output)

cfg = resnet_model.get_config()
cfg['layers'][0]['config']['batch_input_shape'] = (None, input_size, input_size, 1)
model = Model.from_config(cfg)

for i, layer in enumerate(model.layers):
    if i == 1:
        new_weights = resnet_model.layers[i].get_weights()[0].sum(axis=2, keepdims=True)
        model.set_weights([new_weights])
        layer.trainable = False
    elif len(model.layers) - i > 1: #freeze all but last layer
        layer.trainable = False
        layer.set_weights(resnet_model.layers[i].get_weights())
    else:
        layer.trainable = True 
        layer.set_weights(resnet_model.layers[i].get_weights())
        
    
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #10x smaller than standard
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(crop_gen(train_gen),
                              epochs=2,
                              steps_per_epoch=4*len(train_gen), #effectively 1 run through every possibility of reflected data
                              validation_data=crop_gen(test_gen),
                              validation_steps=len(test_gen), 
                              verbose=1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss','val_loss'], loc='upper right');
plt.title('Learning curve for the training of Dense Layers')
plt.show()
print('Final val_acc: '+history.history['val_acc'][-1].astype(str))
from tensorflow.python.keras.optimizers import Adam

for layer in model.layers:
    layer.trainable = True
adam_fine = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #20x smaller than standard
model.compile(optimizer=adam_fine, loss='binary_crossentropy', metrics=['accuracy'])
history2 = model.fit_generator(crop_gen(train_gen),
                              epochs=10,
                              steps_per_epoch=4*len(train_gen), #effectively 1 run through every possibility of reflected data
                              validation_data=crop_gen(test_gen),
                              validation_steps=len(test_gen), 
                              verbose=1)
model.save_weights('ResNet50_weights.h5')
full_history = dict()
for key in history.history.keys():
    full_history[key] = history.history[key]+history2.history[key][1:] #first epoch is wasted due to initialisation of momentum
    
plt.plot(full_history['loss'])
plt.plot(full_history['val_loss'])
plt.legend(['loss','val_loss'], loc='upper right')
plt.title('Full Learning curve for the training process')
plt.show()
print('Final val_acc: '+full_history['val_acc'][-1].astype(str))