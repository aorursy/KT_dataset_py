# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as img
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input, ZeroPadding2D,GlobalAveragePooling2D,MaxPool2D
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.utils import plot_model
import matplotlib.image as mpimg
import os
%system nvidia-smi
%load_ext tensorboard
%tensorboard --logdir logs
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
breed_list = os.listdir("../input/stanford-dogs-dataset/images/Images/")
def show_dir_images(breed, n_to_show):
    plt.figure(figsize=(16,16))
    img_dir = "../input/stanford-dogs-dataset/images/Images/{}/".format(breed)
    images = os.listdir(img_dir)[:n_to_show]
    for i in range(n_to_show):
        img = mpimg.imread(img_dir + images[i])
        plt.subplot(n_to_show/4+1, 4, i+1)
        plt.imshow(img)
        plt.axis('off')
print(breed_list[0])
show_dir_images(breed_list[0], 12)
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
import pathlib

data_dir = data_dir = pathlib.Path("/kaggle/input/stanford-dogs-dataset/images/Images/")
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
CLASS_NAMES
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

datagen = ImageDataGenerator(validation_split=0.1, rescale=1./255)
train_generator = datagen.flow_from_directory("/kaggle/input/stanford-dogs-dataset/images/Images",target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,classes = list(CLASS_NAMES),subset='training',class_mode='categorical',shuffle=True,interpolation = 'lanczos')
valid_generator = datagen.flow_from_directory("/kaggle/input/stanford-dogs-dataset/images/Images",target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,classes = list(CLASS_NAMES),subset='validation',class_mode='categorical',shuffle=True,interpolation = 'lanczos')
batchX, batchy = train_generator.next()
valid_generator.index_array = None
valid_generator.shuffle = False 
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
# for data, labels in train_generator:
#    print(data.shape)  # (64, 200, 200, 3)
#    print(data.dtype)  # float32
#    print(labels.shape)  # (64,)
#    print(labels.dtype)  # int32
image_count = len(list(data_dir.glob('*/*.jpg')))
image_count
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title(),fontsize= 6)
      plt.axis('off')
image_batch, label_batch = next(train_generator)
show_batch(image_batch, label_batch)
print(image_batch.shape)
print(label_batch.shape)
nb_classes = len(train_generator.class_indices)
print(nb_classes)
def buildModel():
    model = Sequential()
    #1 conv layer
    model.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding="valid",activation="relu",input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(BatchNormalization())
    
    #2 conv layer
    model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding="valid",activation="relu"))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(BatchNormalization())
    
    #3 conv layer
    model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))
    
    #4 conv layer
    model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))
    
    #5 conv layer
    model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())

    #1 dense layer
    model.add(Dense(4096,input_shape=(227,227,3),activation="relu"))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    #2 dense layer
    model.add(Dense(4096,activation="relu"))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    #3 dense layer
    model.add(Dense(1000,activation="relu"))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
        

    #output layer
    model.add(Dense(120,activation="softmax"))
    return model
# nb_classes = 120
# def buildModel():
#     model = Sequential()
#     model.add(Conv2D(64,(3,3), padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT,3)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     # 2nd Convolution layer
#     model.add(Conv2D(128,(1,1), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     # 3rd Convolution layer
#     model.add(Conv2D(512,(3,3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     # 4th Convolution layer
#     model.add(Conv2D(512,(3,3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     # Flattening
#     model.add(Flatten())

#     # Fully connected layer 1st layer
#     model.add(Dense(256))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.25))

#     # Fully connected layer 2nd layer
#     model.add(Dense(512))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.25))

#     model.add(Dense(nb_classes, activation='softmax'))
#     return model
model = buildModel().summary()
model = buildModel()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard , CSVLogger, ReduceLROnPlateau
import datetime
checkpoint = ModelCheckpoint("model_vgg.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3,verbose=1)
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir='./logs',histogram_freq=1)
csvlogger = CSVLogger(filename= "training_csv.log", separator = ",", append = False)
callbacks = [checkpoint,early,reduce_lr]

#batch size = 2**x , 16,32,64,24 
epoch = 50
learning_rate = 0.0001
validation_steps = np.ceil(valid_generator.n/BATCH_SIZE)
steps_per_epoch = np.ceil(train_generator.n/BATCH_SIZE)
from keras.optimizers import Adam
model_adam = buildModel()
optimizer = Adam(lr=learning_rate)
model_adam.compile('adam', loss='categorical_crossentropy', metrics=["accuracy"])
history = model_adam.fit_generator(train_generator,epochs=epoch, steps_per_epoch = steps_per_epoch,validation_data=valid_generator, validation_steps=validation_steps,callbacks=callbacks)
acc = history.history['accuracy']
loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize = (16, 5))

plt.subplot(1,2,1)
plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
