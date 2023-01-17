import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2

import glob
train_path = '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/'

test_path = '../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/'



#Finding all the existing categories in training data

categories = np.array([])

for dirs in glob.glob(train_path+'/*'):

    categories = np.append(categories,dirs.split('/')[-1])

print('Classes in the data: ',categories)
#Finding the training set size wrt labels/categories

num_imgs = np.array([])

for i in categories:

    num_imgs = np.append(num_imgs,len(glob.glob(train_path+i+'/*')))

num_imgs = pd.DataFrame([categories,num_imgs],index=['label','no. of images']).T.set_index('label').T

num_imgs
#Plotting some sample pictures for a given label

def plot_samples(label,num_samples=3):

    plt.figure(figsize=(20,6))

    print('Showing sample images of label:',label)

    for i in range(num_samples):

        plt.subplot(1,num_samples,i+1)

        plt.imshow(cv2.imread(glob.glob(train_path+label+'/*')[np.random.randint(0,num_imgs[label][0])]))

    plt.tight_layout()

plot_samples('I',5)
print('Shape of input images:',cv2.imread(glob.glob(train_path+'A/*')[0]).shape)
#Since I, J are similar when rotated, rotation range is limited in datagenerator

plot_samples('J',5)
datagen = ImageDataGenerator(samplewise_center=True,

                            samplewise_std_normalization=True,

                            validation_split=0.1,

                            rotation_range=5,

                            zoom_range=0.1,

                            width_shift_range=0.1,

                            height_shift_range=0.1,

                            fill_mode='nearest')



train_gen = datagen.flow_from_directory(train_path,target_size=(64,64),batch_size=32,shuffle=True,subset='training')

val_gen = datagen.flow_from_directory(train_path,target_size=(64,64),batch_size=32,shuffle=True,subset='validation')
#Plotting some transformed images

plt.figure(figsize=(20,6))

for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(train_gen.next()[0][0])

plt.tight_layout()
from keras.models import Sequential

from keras.layers import Dense,Conv2D,MaxPooling2D, Flatten,Dropout



model = Sequential()



model.add(Conv2D(32,(4,4),strides=1,activation='relu',padding='same', input_shape=(64,64,3)))

model.add(Conv2D(32,(3,3),strides=2,activation='relu',padding='valid'))

model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),strides=1,activation='relu',padding='same'))

model.add(Conv2D(64,(3,3),strides=2,activation='relu',padding='valid'))

model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),strides=1,activation='relu',padding='same'))

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(512,activation='relu'))

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(29,activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
#Fitting the data to the model by using data generator

history = model.fit_generator(train_gen,epochs=15,validation_data=val_gen)
plt.plot(history.history['accuracy'],label='train')

plt.plot(history.history['val_accuracy'],label='validation')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend()
#Finding the prediction data on validation set to plot confusion matrix - helps to analyse errors



validation_gen = datagen.flow_from_directory(train_path,target_size=(64,64),batch_size=1,shuffle=False,subset='validation')

y_pred = np.argmax(model.predict_generator(validation_gen),axis=1)

y_true = validation_gen.classes
import seaborn as sns

from sklearn.metrics import confusion_matrix

plt.figure(figsize=(15,10))

#Here 0 to 28 labels are mapped to their original categories

ax = sns.heatmap(confusion_matrix(y_true,y_pred),annot=True,xticklabels=np.sort(categories),yticklabels=np.sort(categories),cmap='GnBu');

ax.set_xlabel('Predicted values');

ax.set_ylabel('True values');

ax.set_title('Confusion matrix');
#Saving the model weights to load later



model.save_weights("als_hand_sign_model.h5")