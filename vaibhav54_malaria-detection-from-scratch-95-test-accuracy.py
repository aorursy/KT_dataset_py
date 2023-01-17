import cv2                                         

import numpy as np                                  

import os                                          

from random import shuffle                          

from keras.models import Sequential, load_model                 

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten,Activation

from keras.optimizers import Adam

from keras.preprocessing import image              

import matplotlib.pyplot as plt                    

import warnings

warnings.filterwarnings('ignore')

import os

from tqdm import tqdm
path = ('../input/cell-images-for-detecting-malaria/cell_images/cell_images/')

IMG_SIZE = 150,150



data_labels = []

data_images = []
def  create_data(DIR):

     for folder in (os.listdir(DIR)):

            for file in tqdm(os.listdir(os.path.join(DIR,folder))):

                if file.endswith("png"):



    #                 image_names.append(os.path.join(TRAIN_DIR,folder,file))

                    data_labels.append(folder)

                    img = cv2.imread(os.path.join(DIR,folder,file))

                    im = cv2.resize(img,IMG_SIZE)

                    data_images.append(im)

                else:

                    continue
create_data(path)
data_images = np.array(data_images)
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils



from keras.utils import to_categorical





le = LabelEncoder()

label = le.fit_transform(data_labels)

print(label.shape)

encoded = to_categorical(label)

print(encoded.shape)

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



X_train,X_val,y_train,y_val=train_test_split(data_images,encoded,test_size=0.20,random_state=42)







print("X_train shape",X_train.shape)

print("X_test shape",X_val.shape)

print("y_train shape",y_train.shape)

print("y_test shape",y_val.shape)
model = Sequential()



model.add(Conv2D( 32, (3,3), padding='Same', input_shape=(150,150,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))





model.add(Conv2D( 64, (3,3), padding='Same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))





model.add(Conv2D( 128, (3,3), padding='Same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

          

model.add(Flatten())



model.add(Dense(128))          

model.add(Activation('relu'))    



model.add(Dense(32))          

model.add(Activation('relu'))

model.add(Dropout(0.5))





# model.add(Dense(1, activation='sigmoid'))

model.add(Dense(2, activation='softmax'))  

          

model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])



model.summary()
history = model.fit(X_train,y_train,validation_data=(X_val,y_val), batch_size=128, epochs=30, verbose=1)
import matplotlib.pyplot as plt



plt.xlabel('Epochs')

plt.ylabel('Accuracy')



plt.plot(history.history['val_accuracy'],label='val')

plt.plot(history.history['accuracy'],label='train')

plt.legend(('val_accuracy', 'train_accuracy'),

           shadow=True, loc=(0.01, 1), handlelength=1.5, fontsize=10)
plt.plot(history.history['val_loss'],label='val')

plt.plot(history.history['loss'],label='train')

plt.legend(('val_loss', 'train_loss'),shadow=True,

           loc=(0.01, 0.01), handlelength=1.5, fontsize=10)

pred = model.predict_classes(X_val)



y_val=np.argmax(y_val,axis=1)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix





cm = confusion_matrix(pred,y_val)



acc_sc = accuracy_score(pred,y_val)

print('accuracy_score:',acc_sc)

cm
model.save('malaria_scratch.h5')
#TESTING----



from keras.models import load_model





# img_path ='/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144348_cell_125.png'

img_path ='/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_167.png'





img = cv2.imread(i_path)

img = cv2.resize(img,IMG_SIZE)

img = np.expand_dims(img, axis=0)



malaria_class_acc = model.predict(img)

malaria_class = np.argmax(malaria_class_acc[0])



print('Accuracy',malaria_class_acc[0][malaria_class])



if malaria_class==0:

    print('Parasitized')

else:

    print('Uninfected')    