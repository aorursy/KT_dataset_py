import tensorflow as tf
import keras 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
import random
image_size = 2048
input_size = 331
def rand_crop(img):
    h = random.randint(2*input_size, image_size) #2*input_size to prevent cropping too small
    cx = random.randint(0, image_size-h)
    cy = random.randint(0, image_size-h)
    cropped_img = img[cx:cx+h,cy:cy+h,:]
    return cv2.resize(cropped_img, (input_size,input_size))
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
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
data_dir = '../input/neuron cy5 data/Neuron Cy5 Data'

data_gen = ImageDataGenerator(horizontal_flip=True,
                              vertical_flip=True,
                              validation_split=0.2,
                              preprocessing_function = img_transf)
train_gen = data_gen.flow_from_directory(data_dir, 
                                         target_size=(image_size,image_size),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=32, 
                                         shuffle=True, 
                                         subset='training')
test_gen = data_gen.flow_from_directory(data_dir, 
                                         target_size=(image_size, image_size),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=32, 
                                         shuffle=True, 
                                         subset='validation')

classes = dict((v, k) for k, v in train_gen.class_indices.items())
num_classes = len(classes)
def crop_gen(batches):
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.empty((batch_x.shape[0], input_size, input_size, 1))
        for i in range(batch_x.shape[0]):
            batch_crops[i,...,0] = rand_crop(batch_x[i])
        yield (batch_crops, batch_y)
from keras.preprocessing.image import ImageDataGenerator
data_dir = '../input/neuron cy5 data/Neuron Cy5 Data'

data_gen = ImageDataGenerator(horizontal_flip=True,
                              vertical_flip=True,
                              validation_split=0.2,
                              preprocessing_function = img_transf)
train_gen = data_gen.flow_from_directory(data_dir, 
                                         target_size=(image_size,image_size),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=32, 
                                         shuffle=True, 
                                         subset='training')
test_gen = data_gen.flow_from_directory(data_dir, 
                                         target_size=(image_size, image_size),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=32, 
                                         shuffle=True, 
                                         subset='validation')

classes = dict((v, k) for k, v in train_gen.class_indices.items())
num_classes = len(classes)
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.layers import Conv2D, Reshape, MaxPooling2D
from tensorflow.python.keras.models import Model

pretrained_model = VGG16(include_top=False,
                         pooling='none',
                         input_shape=(input_size, input_size, 3),
                         weights='imagenet')
x = MaxPooling2D(pool_size=(10,10), strides=(1,1))(pretrained_model.output)  #pool size chosen as 10x10 to ensure that the output shape is 
x = Conv2D(filters=256, kernel_size=1, activation='relu')(x) #dense layers are replaced with Conv2D to allow different input sizes
x = Conv2D(filters=64, kernel_size=1, activation='relu')(x)
x = Conv2D(filters=num_classes, kernel_size=1, activation='softmax')(x)
outp = Reshape([2])(x) #only for training
vgg16_model = Model(pretrained_model.input, outp)

cfg = vgg16_model.get_config()
cfg['layers'][0]['config']['batch_input_shape'] = (None, None, None, 1) #any size image can be input
model = Model.from_config(cfg)
  
for i, layer in enumerate(model.layers):
    if i == 1:
        new_weights = np.reshape(vgg16_model.layers[i].get_weights()[0].sum(axis=2),(3,3,1,64))
        model.set_weights([new_weights])
        layer.trainable = False
    elif len(model.layers) - i > 3: #freeze all but last 3 layers
        layer.trainable = False
        layer.set_weights(vgg16_model.layers[i].get_weights())
    else:
        layer.trainable = True 
        layer.set_weights(vgg16_model.layers[i].get_weights())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(crop_gen(train_gen),
                              epochs=10,
                              steps_per_epoch=4*len(train_gen), #effectively 1 run through every possibility of reflected data
                              validation_data=crop_gen(test_gen),
                              validation_steps=len(test_gen), 
                              verbose=1)
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
full_history = dict()
for key in history.history.keys():
    full_history[key] = history.history[key]+history2.history[key][1:] #first epoch is wasted due to initialisation of momentum
    
plt.plot(full_history['loss'])
plt.plot(full_history['val_loss'])
plt.legend(['loss','val_loss'], loc='upper right')
plt.title('Full Learning curve for the training process')
plt.show()
print('Final val_acc: '+full_history['val_acc'][-1].astype(str))
swmodel = Model(model.input, model.layers[-2].output)
swmodel.summary()
def lcm(a,b): 
    from math import gcd
    return (a*b)//gcd(a,b)
X_test, y_test = test_gen.next()
y_pred_maps = swmodel.predict(X_test, batch_size=1, verbose=1)
y_pred_var = y_pred_maps[...,0].var(axis=(1,2))
idx = np.argmax(y_pred_var)
uns_img = X_test[idx,...,0]
uns_img = np.uint8(255*uns_img)
uns_img = cv2.cvtColor(uns_img,cv2.COLOR_GRAY2RGB)
heatmap = y_pred_maps[idx,...,0] #This heatmap will give high values for treated areas
heatmap /= heatmap.max() #For visulisation
heatmap = np.pad(heatmap, mode='constant', pad_width=1, constant_values=0.5)
heatmap = np.uint8(255*cv2.resize(heatmap, (image_size, image_size)))
heatmap = cv2.applyColorMap(255-heatmap, cv2.COLORMAP_JET) #255-heatmap so that red is high values
superimposed_map = cv2.addWeighted(uns_img, 0.6, heatmap, 0.4, 0)
plt.imshow(superimposed_map)
plt.title('Classification map for most uncertain prediction:')
plt.show()
print('Actual class: '+classes[y_test[idx,0]])
heatmap = y_pred_maps[idx,...,0] #This heatmap will give high values for treated areas
heatmap /= heatmap.max() #For visulisation
heatmap = np.uint8(255*cv2.resize(heatmap, (image_size, image_size)))
pooled_heatmap = np.empty(heatmap.shape)
heatmap = np.pad(heatmap, mode='reflect', pad_width=150)
for i in range(pooled_heatmap.shape[0]):
    percent = i*100/pooled_heatmap.shape[0]
    prog = ' Pooling: ' +str(percent) +'%';
    sys.stdout.write('\r'+prog)
    for j in range(pooled_heatmap.shape[1]):
        pooled_heatmap[i,j] = np.mean(heatmap[i:i+331, j:j+331])
sys.stdout.write('\rDone                   ')
pooled_heatmap = np.uint8(pooled_heatmap.round())
pooled_heatmap = cv2.applyColorMap(255-pooled_heatmap, cv2.COLORMAP_JET) #255-heatmap so that red is high values
superimposed_map = cv2.addWeighted(uns_img, 0.6, pooled_heatmap, 0.4, 0)
plt.imshow(superimposed_map)
plt.title('Pooled classification map for most uncertain prediction:');