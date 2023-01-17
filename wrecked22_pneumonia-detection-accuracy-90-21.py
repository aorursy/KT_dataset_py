import numpy as np 

import matplotlib.pyplot as plt 

import os

from PIL import Image

print(os.listdir("../input"))

import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator, load_img

import tensorflow as tf
print(os.listdir('../input/chest_xray/chest_xray'))
folder_train_pneu = '../input/chest_xray/chest_xray/train/PNEUMONIA/'

folder_train_norm = '../input/chest_xray/chest_xray/train/NORMAL/'



folder_test_pneu = '../input/chest_xray/chest_xray/test/PNEUMONIA/'

folder_test_norm = '../input/chest_xray/chest_xray/test/NORMAL/'



Val = '../input/chest_xray/chest_xray/val/PNEUMONIA/'

Val = '../input/chest_xray/chest_xray/val/NORMAL/'
#Normal

print(len(os.listdir(folder_train_norm)))

rand_norm= np.random.randint(0,len(os.listdir(folder_train_norm)))

norm_pic = os.listdir(folder_train_norm)[rand_norm]

print('normal picture title: ',norm_pic)



norm_pic_address = folder_train_norm+norm_pic



#Pneumonia

rand_p = np.random.randint(0,len(os.listdir(folder_train_pneu)))



sic_pic =  os.listdir(folder_train_pneu)[rand_norm]

sic_address = folder_train_pneu+sic_pic

print('pneumonia picture title:', sic_pic)
norm_load = Image.open(norm_pic_address)

sic_load = Image.open(sic_address)



f = plt.figure(figsize= (10,6))

a1 = f.add_subplot(1,2,1)

img_plot = plt.imshow(norm_load)

a1.set_title('Normal')



a2 = f.add_subplot(1, 2, 2)

img_plot = plt.imshow(sic_load)

a2.set_title('Pneumonia')
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train',

                                                              target_size = (100, 100),

                                                              batch_size = 32,

                                                              class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory('../input/chest_xray/chest_xray/val/',

                                                              target_size=(100, 100),

                                                              batch_size=32,

                                                              class_mode='binary')



test_set = test_datagen.flow_from_directory('../input/chest_xray/chest_xray/test',

                                                              target_size = (100, 100),

                                                              batch_size = 32,

                                                              class_mode = 'binary')
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(128,(3,3), input_shape = (100,100,3) ,activation = tf.nn.relu ))

model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=None))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(32,activation=tf.nn.relu))

model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.summary

cnn_model = model.fit_generator(training_set,

                                steps_per_epoch = 163,

                                epochs = 10,

                                validation_data = validation_generator,

                                validation_steps = 624)
test_accu = model.evaluate_generator(test_set,steps=624)
print('The testing accuracy is :',test_accu[1]*100, '%')
plt.plot(cnn_model.history['acc'])

plt.plot(cnn_model.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
plt.plot(cnn_model.history['val_loss'])

plt.plot(cnn_model.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()
CNN_model = tf.keras.models.Sequential()

CNN_model.add(tf.keras.layers.Conv2D(128,(3,3),input_shape=(100,100,3), activation=tf.nn.relu,padding="valid"))

CNN_model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=None))



CNN_model.add(tf.keras.layers.Conv2D(128,(3,3), activation=tf.nn.relu, padding="same"))

CNN_model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=None))

CNN_model.add(tf.keras.layers.Flatten())



CNN_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

CNN_model.add(tf.keras.layers.Dropout(0.25))



CNN_model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))

CNN_model.add(tf.keras.layers.Dropout(0.35))



CNN_model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
CNN_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
Model = CNN_model.fit_generator(training_set,     steps_per_epoch = 163,

                                                  epochs = 10,

                                                  validation_data = validation_generator,

                                                  validation_steps = 624)
test_accu = CNN_model.evaluate_generator(test_set,steps=624)

print('The testing accuracy is :',test_accu[1]*100, '%')
# from mpl_toolkits.mplot3d import Axes3D

# from sklearn.preprocessing import StandardScaler

# import os

# import matplotlib.pyplot as plt

# import numpy as np

# import pandas as pd

# import os

# from glob import glob

# import seaborn as sns

# from PIL import Image

# from sklearn.svm import SVC

# from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import accuracy_score

# import glob

# import cv2





# folder_train_pneu = '../input/chest_xray/chest_xray/train/PNEUMONIA'

# folder_train_norm = '../input/chest_xray/chest_xray/train/NORMAL'



# folder_test_pneu = '../input/chest_xray/chest_xray/test/PNEUMONIA'

# folder_test_norm = '../input/chest_xray/chest_xray/test/NORMAL'



# Val = '../input/chest_xray/chest_xray/val/PNEUMONIA'

# Val = '../input/chest_xray/chest_xray/val/NORMAL'





# X_test_normal = [cv2.imread(file) for file in glob.glob("../input/chest_xray/chest_xray/test/NORMAL/*.jpeg")]

# X_test_pneu = [cv2.imread(file) for file in glob.glob("../input/chest_xray/chest_xray/test/PNEUMONIA/*.jpeg")]

# X_train_normal = [cv2.imread(file) for file in glob.glob("../input/chest_xray/chest_xray/train/NORMAL/*.jpeg")]

# X_train_pneu = [cv2.imread(file) for file in glob.glob("../input/chest_xray/chest_xray/train/PNEUMONIA/*.jpeg")]



# plt.figure()

# plt.imshow(X_test_normal[1]) 

# plt.show()



# test_Normal = get_image_files("../input/chest_xray/chest_xray/test/NORMAL")

# test_Pneumonia = get_image_files("../input/chest_xray/chest_xray/test/PNEUMONIA")

# train_Normal = get_image_files("../input/chest_xray/chest_xray/train/NORMAL")

# train_Pneumonia = get_image_files("../input/chest_xray/chest_xray/train/PNEUMONIA")

# from subprocess import check_output

# print(check_output(["ls", "../input/chest_xray/chest_xray/test"]).decode("utf8"))



# y_n_test = np.zeros((np.shape(test_Normal)))

# y_p_train = np.ones((np.shape(train_Pneumonia)))

# y_n_train = np.zeros((np.shape(train_Normal)))

# y_p_test = np.ones((np.shape(test_Pneumonia)))



# print(y_n_train.shape,y_p_train.shape)



# X_train = np.concatenate((train_Pneumonia, train_Normal), axis = 0)

# y_train = np.concatenate((y_p_train, y_n_train), axis = 0)

# X_test = np.concatenate((test_Pneumonia, test_Normal), axis = 0)

# y_test = np.concatenate((y_p_test, y_n_test), axis = 0)



# s = np.arange(X_train.shape[0])

# np.random.shuffle(s)

# X_train = X_train[s]

# y_train = y_train[s]



# s = np.arange(X_test.shape[0])

# np.random.shuffle(s)

# X_test = X_test[s]

# y_test = y_test[s]