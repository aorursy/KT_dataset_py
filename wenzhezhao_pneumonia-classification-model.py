import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from glob import glob

from PIL import Image

%matplotlib inline

import matplotlib.pyplot as plt

import cv2

import fnmatch

import keras

from time import sleep

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization,MaxPooling2D,Activation

from keras.optimizers import RMSprop,Adam

from tensorflow.keras.callbacks import EarlyStopping

from keras import backend as k

from keras.applications.vgg16 import VGG16
print(os.listdir("../input/ai-yanxi"))
train = pd.read_csv("../input/ai-yanxi/train.csv",header=None)

train.columns = ['ID', 'Category']



#train_bboxes = pd.read_csv("../input/ai-yanxi/train_bboxes.csv")

path_train = "../input/ai-yanxi/train/"

path_test = "../input/ai-yanxi/test/"

total_images_train = os.listdir(path_train)

total_images_test = os.listdir(path_test)

train["dir"]=path_train+train["ID"].map(str)+".jpg"
train.head()
# imagePatches = glob('../input/ai-yanxi/train/*.jpg', recursive=True)

# print(len(imagePatches))

# os.path.splitext(imagePatches[0])[1]

# imagePatches

### show 8547.jpeg,7653.jpeg,...(which are location of pneumonia figure)

# print(total_images_train[0])

# len(total_images_train)
#train["dir"]=path_train+train["ID"].map(str)+".jpg"
image = cv2.imread(path_train+"5.jpg")

plt.imshow(image)

print(image.shape)
plt.imshow(image[:,:,2])
# Get few samples for  the classes

class0_samples = (train[train['Category']==0]['dir'].iloc[:5]).tolist()

class1_samples = (train[train['Category']==1]['dir'].iloc[:5]).tolist()

class2_samples = (train[train['Category']==2]['dir'].iloc[:5]).tolist()

class3_samples = (train[train['Category']==3]['dir'].iloc[:5]).tolist()

class4_samples = (train[train['Category']==4]['dir'].iloc[:5]).tolist()



# Concat the data in a single list and del the above two list

samples = class0_samples + class1_samples + class2_samples + class3_samples + class4_samples

del class0_samples , class1_samples , class2_samples , class3_samples , class4_samples



# Plot the data 

f, ax = plt.subplots(5,5, figsize=(30,25))

for i in range(25):

    img = plt.imread(samples[i])

    ax[i//5, i%5].imshow(img, cmap='gray')

    if i<5:

        ax[i//5, i%5].set_title("class0")

    elif i<10:

        ax[i//5, i%5].set_title("class1")

    elif i<15:

        ax[i//5, i%5].set_title("class2")

    elif i<20:

        ax[i//5, i%5].set_title("class3")

    else:

        ax[i//5, i%5].set_title("class4")

    ax[i//5, i%5].axis('off')

    ax[i//5, i%5].set_aspect('auto')

plt.show()
x = []

for i in range(train.shape[0]):

    full_size_image = cv2.imread(path_train+str(i)+".jpg")

    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)

    x.append(im)

    if i%2500 == 0:

        print(i)

x = np.array(x)

y = np.array(train["Category"])

y=to_categorical(y, num_classes = 5)
import keras

from keras.models import Sequential,Input,Model

from keras.layers import InputLayer,Conv2D, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, LSTM, TimeDistributed

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU 

model = Sequential()

#model.add(InputLayer(input_shape=(224,224,3)))

model.add(Conv2D(32,(7,7),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.15))

model.add(Conv2D(64,(5,5),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.15))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.15))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.15))

model.add(GlobalAveragePooling2D())

model.add(Dense(1000, activation='relu'))

model.add(Dense(5,activation='softmax'))

model.build(input_shape=(None,224,224,3))

model.summary()

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.05, random_state = 101)

# print(y_train.shape)

# print(y_train.shape)

# del x, y
from keras.callbacks import ModelCheckpoint

mcp = ModelCheckpoint(filepath='model_check_path.hdf5',monitor="val_accuracy", save_best_only=True, save_weights_only=False)

#hist = model.fit(x_train,y_train,batch_size = 32, epochs = 20, verbose=1,  validation_split=0.2)

hist = model.fit(x_train,y_train,batch_size = 64, epochs = 20, verbose=1,validation_data=(x_valid,y_valid),callbacks=[mcp])
print(hist.history.keys())
model.load_weights("model_check_path.hdf5")     ######################### 含泪重要！！！！
model.evaluate(x_valid,y_valid)
fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_facecolor('w')

ax.grid(b=False)

ax.plot(hist.history['accuracy'], color='red')

ax.plot(hist.history['val_accuracy'], color ='green')

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower right')

plt.show()
x_test = []

for i in range(len(total_images_test)):

    full_size_image = cv2.imread(path_test+str(i)+".jpg")

    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)

    x_test.append(im)

    if len(x_test) % 1000 == 0:

        print(len(x_test))

x_test = np.array(x_test)
print(x_test.shape)
predictions = model.predict(x_test)
predict=np.argmax(predictions, axis=1)
# idpre = pd.DataFrame({

#     'Id':total_images_test,

#     'pre':predict

# })

idpre = pd.DataFrame({

    "ID":np.arange(len(total_images_test)),

    'pre':predict

})

idpre.to_csv('idpre4.csv', index = False, header = False)
# from keras.applications.vgg16 import VGG16

# from keras.preprocessing import image

# from keras.applications.vgg16 import preprocess_input

# from keras.layers import Input, Flatten, Dense

# from keras.models import Model

# import numpy as np



# #Get back the convolutional part of a VGG network trained on ImageNet

# model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

# model_vgg16_conv.summary()



# #Create your own input format (here 224x224x3)

# input = Input(shape=(224,224,3),name = 'image_input')



# #Use the generated model 

# output_vgg16_conv = model_vgg16_conv(input)



# #Add the fully-connected layers 

# point = Flatten(name='flatten')(output_vgg16_conv)

# point = Dense(4096, activation='relu', name='fc1')(point)

# point = Dense(4096, activation='relu', name='fc2')(point)

# point = Dense(5, activation='softmax', name='predictions')(point)



# #Create your own model 

# my_model = Model(input=input, output=point)



# #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training

# my_model.summary()

# #Then training with your data ! 

# my_model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
# from keras.callbacks import ModelCheckpoint

# check = ModelCheckpoint(filepath='model_vg_check_path.hdf5',monitor="val_accuracy", save_best_only=True, save_weights_only=False)

# #hist = model.fit(x_train,y_train,batch_size = 32, epochs = 20, verbose=1,  validation_split=0.2)

# # hist = my_model.fit(x,y,batch_size = 64, epochs = 5, verbose=1, validation_split=0.1,callbacks=[check])

# hist = my_model.fit(x_train,y_train,batch_size = 64, epochs = 10, verbose=1, validation_data=(x_valid,y_valid),callbacks=[check])
# model = Sequential()

# model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Flatten())

# model.add(Dense(units=4096,activation="relu"))

# model.add(Dense(units=4096,activation="relu"))

# model.add(Dense(units=5, activation="softmax"))

# model.summary()

# model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])