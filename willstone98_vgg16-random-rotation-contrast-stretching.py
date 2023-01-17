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
data_dir = '../input/neuron cy5 full/Neuron Cy5 Full'
folders = {'Treated': 1, 'Untreated': 0}
tags = {'B':0, 'C':36, 'D':72, 'E':108, 'F':144, 'G':180, 'AB':216, 'AC':246, 'AD':276, 'AE':307, 'AF':337, 'AG':367}
img_dirs = [None]*792
y = np.empty([792,2])

image_size = 2048

for folder in os.listdir(data_dir):
    folder_dir = os.path.join(data_dir, folder)
    bin_class = folders[folder]
    for idx, file in enumerate(os.listdir(folder_dir)):
        img_dirs[idx+bin_class*396] = os.path.join(folder_dir, file)
        y[idx+bin_class*396,:] = [1-bin_class,bin_class]

def imload(file_dir):
    img = cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE)
    return img  

num_pixels = image_size**2

pix_bins = np.zeros(256)
print('Calculating image intensities:')
for i, img_dir in enumerate(img_dirs):
    im = imload(img_dirs[idx])
    pix_bins = np.add(np.bincount(im.ravel(), minlength=256), pix_bins)
    prog = 'Progress: '+str(i+1)+'/'+str(len(img_dirs))
    sys.stdout.write('\r'+prog)
sys.stdout.write('\r Done            \n')
cdf = pix_bins.cumsum()
cdf *= 255/cdf.max()
p1 = np.argmin(np.abs(cdf-0.01*255))
p99 = np.argmin(np.abs(cdf-0.99*255))
def img_transf(img):
    img -= p1
    img /= p99
    img = 255*np.clip(img, 0, 1)
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
from tensorflow.python.keras.applications import VGG19
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling2D
from tensorflow.python.keras.optimizers import Adam

pretrained_model = VGG19(include_top=False,
                         pooling='none',
                         input_shape=(input_size, input_size, 3),
                         weights='imagenet')
x = GlobalMaxPooling2D()(pretrained_model.output)
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)  
vgg16_model = Model(pretrained_model.input, output)

cfg = vgg16_model.get_config()
cfg['layers'][0]['config']['batch_input_shape'] = (None, input_size, input_size, 1)
model = Model.from_config(cfg)

for i, layer in enumerate(model.layers):
    if i == 1:
        new_weights = vgg16_model.layers[i].get_weights()[0].sum(axis=2, keepdims=True)
        model.set_weights([new_weights])
        layer.trainable = False
    elif len(model.layers) - i > 3: #freeze all but last 3 layers
        layer.trainable = False
        layer.set_weights(vgg16_model.layers[i].get_weights())
    else:
        layer.trainable = True 
        layer.set_weights(vgg16_model.layers[i].get_weights())
           
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #10x smaller than standard
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
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
adam_fine = Adam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #50x smaller than standard
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