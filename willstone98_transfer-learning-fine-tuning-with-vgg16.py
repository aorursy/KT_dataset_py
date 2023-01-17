import tensorflow as tf
import keras 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
import random
from keras.preprocessing.image import ImageDataGenerator#
image_size = 1024
input_size = 331
train_dir = '../input/neuron cy5 train data/Neuron Cy5 Train Data'
test_dir = '../input/neuron cy5 test data/Neuron Cy5 Test Data'

# Create generator with augmentation for training
train_gen = ImageDataGenerator(horizontal_flip=True,
                               vertical_flip=True,
                               samplewise_center=True,
                               samplewise_std_normalization=True).flow_from_directory(train_dir, 
                                                                                      target_size=(image_size,image_size),
                                                                                      color_mode='grayscale',
                                                                                      class_mode='categorical',
                                                                                      batch_size=16,
                                                                                      shuffle=True)
# Create generator with no augmentation for testing
test_gen = ImageDataGenerator(samplewise_center=True,
                              samplewise_std_normalization=True).flow_from_directory(test_dir, 
                                                                                     target_size=(input_size,input_size),
                                                                                     color_mode='grayscale',
                                                                                     class_mode='categorical',
                                                                                     batch_size=16,
                                                                                     shuffle=True)
classes = dict((v, k) for k, v in train_gen.class_indices.items())
num_classes = len(classes)
def rand_crop(img):
    # Select size of crop
    h = random.randint(input_size, image_size) 
    # Select lower left corner for crop
    cx = random.randint(0, image_size-h)
    cy = random.randint(0, image_size-h)
    # Crop and resize image to input_size
    cropped_img = img[cx:cx+h,cy:cy+h,:]
    return cv2.resize(cropped_img, (input_size,input_size))
def crop_gen(batch_gen):
    '''Generator of cropped images.
    Takes a generator as it's input and returns cropped versions of the generators output.    
    '''
    while True:
        batch_x, batch_y = next(batch_gen)
        batch_crops = np.zeros((batch_x.shape[0], input_size, input_size, 1))
        for i in range(batch_x.shape[0]):
            batch_crops[i,...,0] = rand_crop(batch_x[i])
        yield (batch_crops, batch_y)
from keras import backend as K
    
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import VGG19
from tensorflow.python.keras.layers import GlobalMaxPooling2D, Dense
from tensorflow.python.keras.optimizers import Adam

# Create a VGG19 architecture
pretrained_model = VGG19(include_top=False,
                         pooling='none',
                         input_shape=(input_size, input_size, 3),
                         weights='imagenet')
x = GlobalMaxPooling2D()(pretrained_model.output)
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)
vgg19_model = Model(pretrained_model.input, output)

# Create new model with modified config which accepts the input shape: [input_size, input_size, 1]
cfg = vgg19_model.get_config()
cfg['layers'][0]['config']['batch_input_shape'] = (None, input_size, input_size, 1)
model = Model.from_config(cfg)

# Set the model weights to that of the pretrained VGG19
# and freeze all layers except the final Dense layers.
for i, layer in enumerate(model.layers):
    if i == 1:
        new_weights = np.reshape(vgg19_model.layers[i].get_weights()[0].sum(axis=2),(3,3,1,64))
        model.set_weights([new_weights])
        layer.trainable = False
    elif len(model.layers) - i > 3: # Freeze all but last 3 layers
        layer.trainable = False
        layer.set_weights(vgg19_model.layers[i].get_weights())
    else:
        layer.trainable = True 
        layer.set_weights(vgg19_model.layers[i].get_weights())

# Compile the model with Adam optimizer, binary crossentropy loss and f1 score as the metric
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # 10x smaller than standard
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[f1])
model.summary()
# Train the model
history = model.fit_generator(crop_gen(train_gen),
                              epochs=5,
                              steps_per_epoch=4*len(train_gen), # Effectively 1 run through every possibility of reflected data
                              validation_data=test_gen,
                              validation_steps=len(test_gen), 
                              verbose=1)
# Plot learning curve
plt.figure(figsize=(15,15))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss','val_loss'], loc='upper right',prop={'size': 15});
plt.title('Learning curve for the training of Dense Layers', fontsize=15)
plt.show()
print('Best test F1 score: ' + max(history.history['val_f1']).astype(str))
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint

# Unfreeze all weights
for layer in model.layers:
    layer.trainable = True
    
# Recompile with Adam optimizer with smaller initial learning rate
adam_fine = Adam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # 50x smaller than standard
checkpoint = ModelCheckpoint('VGG19_weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0)

model.compile(optimizer=adam_fine, loss='binary_crossentropy', metrics=[f1])
# Train the model
history2 = model.fit_generator(crop_gen(train_gen),
                              epochs=10,
                              steps_per_epoch=4*len(train_gen), #effectively 1 run through every possibility of reflected data
                              validation_data=test_gen,
                              validation_steps=len(test_gen), 
                              callbacks =[checkpoint],
                              verbose=1)
# Plot learning curve for the whole training process
full_history = dict()
for key in history.history.keys():
    full_history[key] = history.history[key]+history2.history[key]

plt.figure(figsize=(15,15))
plt.plot(full_history['loss'])
plt.plot(full_history['val_loss'])
plt.legend(['loss','val_loss'], loc='upper right',prop={'size': 15});
plt.title('Learning curve for the full training process', fontsize=15)
plt.show()
print('Best test F1 score: ' + max(full_history['val_f1']).astype(str))