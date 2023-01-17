import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

import pandas as pd

from sklearn import metrics

import matplotlib.pyplot as plt

import os



import glob
#Load dataset



PATH='/kaggle/input/chest-xray-pneumonia/chest_xray/'



train_dir = os.path.join(PATH,'train')

# I will use test set for validation

validation_dir = os.path.join(PATH,'test')



train_normal_dir = os.path.join(train_dir, 'NORMAL')  

train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA') 



validation_normal_dir = os.path.join(validation_dir, 'NORMAL')

validation_pneumonia_dir = os.path.join(validation_dir, 'PNEUMONIA')



num_normal_tr = len(os.listdir(train_normal_dir))

num_pneumonia_tr = len(os.listdir(train_pneumonia_dir))



num_normal_val = len(os.listdir(validation_normal_dir))

num_pneumonia_val = len(os.listdir(validation_pneumonia_dir))



total_train = num_normal_tr + num_pneumonia_tr

total_val = num_normal_val + num_pneumonia_val



print('total training normal images:', num_normal_tr)

print('total training pneumonia images:', num_pneumonia_tr)



print('total validation normal images:', num_normal_val)

print('total validation pneumonia images:', num_pneumonia_val)

print("--")

print("Total training imag:", total_train)

print("Total validation images:", total_val)
#global parameters

batch_size = 8

epochs = 4

IMG_HEIGHT = 240

IMG_WIDTH = 240
#Let's create train generator

img_generator_train = ImageDataGenerator(rescale=1./255,

                                         horizontal_flip=True,

                                         rotation_range=45,

                                         zoom_range=0.5,

                                         width_shift_range=0.15,

                                         height_shift_range=0.15)



img_data_gen = img_generator_train.flow_from_directory(directory=train_dir,

                                                       target_size=(IMG_HEIGHT,IMG_WIDTH),

                                                       batch_size=batch_size,

                                                       class_mode='binary')
#Validation generator

validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory=validation_dir,

                                                             shuffle=True, target_size=(IMG_HEIGHT,IMG_WIDTH),

                                                             class_mode='binary')
#Let's have a look on train samples with augmentation

samples,_ = next(img_data_gen) 

plt.figure(figsize=(14,5))

for n,im in enumerate(samples[:5]):

    plt.subplot(1,5,n+1)

    plt.imshow(im)
!pip install git+https://github.com/qubvel/efficientnet
# I like EfficientNet and will tune this model on X-ray dataset



import keras

import efficientnet.keras as efn 



from keras.models import Model, load_model

from keras.layers import Dense

from keras import backend as K
def get_model():

        K.clear_session()

        base_model =  efn.EfficientNetB2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

        x = base_model.output

        x = keras.layers.GlobalAveragePooling2D()(x)

        y_pred = Dense(1, activation='sigmoid')(x)

        return Model(inputs=base_model.input, outputs=y_pred)
model = get_model()
#Model Summury

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(img_data_gen, steps_per_epoch=total_train//batch_size,

                              epochs=epochs,

                              validation_data=val_data_gen,

                              validation_steps=total_val//batch_size)

model.save('efficientNet_240_240_demo')
from keras.models import load_model

model = load_model('efficientNet_240_240_demo')
# Results interpritation
steps = len(val_data_gen.filenames)//batch_size



test_image_generator = ImageDataGenerator(rescale=1./255)



test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size, directory=validation_dir,

                                                             shuffle=False, target_size=(IMG_HEIGHT,IMG_WIDTH),

                                                             class_mode='binary')



preds = model.predict_generator(test_data_gen, steps=steps, verbose=1)

res = {k:(v[0],1 if k.split('/')[0]=='PNEUMONIA' else 0) for k,v in zip(test_data_gen.filenames,preds)}
res
y = [res[i][1] for i in res]

pred = [res[i][0]>=.5 for i in res]
acc = metrics.accuracy_score(y, pred)

recall = metrics.recall_score(y, pred)

precision = metrics.precision_score(y, pred)



(tn, fp), (fn, tp) = metrics.confusion_matrix(y, pred)



print(f'Accuracy: {acc}')

print(f'Recall: {recall}')

print(f'Precision: {precision}')

print(f'True negativ: {tn}, False positive: {fp} ')

print(f'False negativ: {fn}, True positive: {tp} ')