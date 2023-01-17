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
def img_transf(img):
    img -= img.mean()
    img /= np.maximum(img.std(), 1/image_size**2) #prevent /0
    return img
from skimage.transform import rotate
def rand_crop(img):
    img /= 255
    theta = 5*random.randint(0,9)
    img = rotate(img,theta, resize=False)
    max_height = np.floor(image_size/(np.cos(np.pi*theta/180)+np.sin(np.pi*theta/180)))
    min_border = np.ceil((image_size-max_height)/2)
    h = random.randint(input_size, max_height) 
    cx = random.randint(min_border, min_border+max_height-h)
    cy = random.randint(min_border, min_border+max_height-h)
    cropped_img = img[cx:cx+h,cy:cy+h,...]
    return cv2.resize(cropped_img, (input_size,input_size))
from keras.preprocessing.image import ImageDataGenerator
data_dir = '../input/neuron cy5 full/Neuron Cy5 Full'

data_gen = ImageDataGenerator(horizontal_flip=True,
                              vertical_flip=True,
                              validation_split=0.2,
                              preprocessing_function = img_transf)
train_gen = data_gen.flow_from_directory(data_dir, 
                                         target_size=(image_size,image_size),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=16, 
                                         subset='training',
                                         shuffle=True)
test_gen = data_gen.flow_from_directory(data_dir, 
                                        target_size=(image_size, image_size),
                                        color_mode='grayscale',
                                        class_mode='categorical',
                                        batch_size=16, 
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
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling2D


pretrained_model = ResNet50(include_top=False,
                         pooling='none',
                         input_shape=(input_size, input_size, 3),
                        weights='imagenet')
cfg = pretrained_model.get_config()
cfg['layers'][0]['config']['batch_input_shape'] = (None, input_size, input_size, 1)
resnet_model = Model.from_config(cfg)
for i, layer in enumerate(resnet_model.layers):
    if i == 1:
        new_weights = pretrained_model.layers[i].get_weights()[0].sum(axis=2, keepdims=True)
        resnet_model.set_weights([new_weights])
        layer.trainable = False
    else: 
        layer.set_weights(pretrained_model.layers[i].get_weights())
        layer.trainable = False

x = GlobalMaxPooling2D()(resnet_model.output)
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
outp = Dense(num_classes, activation='softmax')(x)
model = Model(resnet_model.input, outp)        
    
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(crop_gen(train_gen),
                              epochs=5,
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
                              epochs=20,
                              steps_per_epoch=4*len(train_gen), #effectively 1 run through every possibility of reflected data
                              validation_data=crop_gen(test_gen),
                              validation_steps=len(test_gen), 
                              verbose=1)
full_history = dict()
for key in history.history.keys():
    full_history[key] = history.history[key]+history2.history[key][1:] #first epoch is wasted due to initialisation of momentum
    
plt.plot(full_history['loss'])
plt.plot(full_history['val_loss'])
plt.legend(['loss','val_loss'], loc='upper right')
plt.title('Full Learning curve for the training process')
plt.show()
print('Final val_acc: '+full_history['val_acc'][-1].astype(str))