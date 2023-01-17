import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,BatchNormalization,Conv2D,MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from glob import glob
from skimage.transform import resize
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix,classification_report
import itertools
train_dir="../input/american-sign-language-recognition/training_set"
test_dir="../input/american-sign-language-recognition/test_set"
folders=[folder for folder in sorted(os.listdir(train_dir)) if folder!='fuck you']
print(folders)
print("Total no. of folders are: ",len(folders))
map_characters={0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '1', 27: '2', 28: '3', 29:'4',30:'5',31:'6',32:'7',33:'8',34:'9',35:'10',36:'best of luck',37:'i love you',38:'space'}
map_characters
from IPython.display import Image 

count=0
for folder in folders:
    count=0
    folder_path=os.path.join(train_dir,folder)
    for image in os.listdir(folder_path):
        image_path=os.path.join(folder_path,image)
        pil_img = Image(filename=image_path)
        display(pil_img)
        count+=1
        break
def load_data(train_dir):
    images=list()
    labels=list()
    for file in folders:
        folder_path=os.path.join(train_dir,file)
        for img in tqdm(os.listdir(folder_path)):
            if file=='A':
                label=0
            elif file=='B':
                label=1
            elif file=='C':
                label=2
            elif file=='D':
                label=3
            elif file=='E':
                label=4
            elif file=='F':
                label=5
            elif file=='G':
                label=6
            elif file=='H':
                label=7
            elif file=='I':
                label=8
            elif file=='J':
                label=9
            elif file=='K':
                label=10
            elif file=='L':
                label=11
            elif file=='M':
                label=12
            elif file=='N':
                label=13
            elif file=='O':
                label=14
            elif file=='P':
                label=15
            elif file=='Q':
                label=16
            elif file=='R':
                label=17
            elif file=='S':
                label=18
            elif file=='T':
                label=19
            elif file=='U':
                label=20
            elif file=='V':
                label=21
            elif file=='W':
                label=22
            elif file=='X':
                label=23
            elif file=='Y':
                label=24
            elif file=='Z':
                label=25
            elif file=='1':
                label=26
            elif file=='2':
                label=27
            elif file=='3':
                label=28
            elif file=='4':
                label=29
            elif file=='5':
                label=30
            elif file=='6':
                label=31
            elif file=='7':
                label=32
            elif file=='8':
                label=33
            elif file=='9':
                label=34
            elif file=='10':
                label=35
            elif file=='best of luck':
                label=36
            elif file=='i love you':
                label=37
            elif file=='space':
                label=38
            if img is not None:
                image_path=folder_path+'/'+img
                image=cv2.imread(image_path)
                image=cv2.resize(image,(64,64))
                gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                images.append(gray)
                labels.append(label)
    images=np.asarray(images)
    labels=np.asarray(labels)
            
    return (images,labels)

X,y=load_data(train_dir)
X_test,y_test=load_data(test_dir)
print('There are total {} Images and {} unique Labels in the dataset.'.format(len(X),len(np.unique(y))))
X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2)
X_train=X_train.reshape(47068,64,64,1)
print(X_train.shape)
X_valid=X_valid.reshape(11768,64,64,1)
X_valid.shape
y_train_vectors=to_categorical(y_train)
y_valid_vectors=to_categorical(y_valid)
y_test_vectors=to_categorical(y_test)
y_train_vectors.shape
X_train=X_train.astype('float32')/255.0
X_valid=X_valid.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0
model=Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64,64, 1))) 
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

    # second conv layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

    # flatten and put a fully connected layer
model.add(Flatten())
model.add(Dense(128, activation='relu')) # fully connected
model.add(Dropout(0.5))

    # softmax layer
model.add(Dense(39, activation='softmax'))
optimiser = Adam()
model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model.summary())
    
history=model.fit(X_train, y_train_vectors,
          batch_size=128,
          epochs=12,
          verbose=1,
          validation_data=(X_valid,y_valid_vectors))
    

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy of vgg16')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
X_test=X_test.reshape(7800,64,64,1)
predictions=model.predict_classes(X_test)
print(classification_report(y_test, predictions))
model.save('Final_Model.h5')