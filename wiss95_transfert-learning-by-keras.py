import numpy as np

import cv2

import matplotlib.pyplot as plt

import os

import random

from tqdm import tqdm

import gc



data='../input/my-data/train'



categories=['Plaque','Fausse plaque']



training_data=[]

def create_training_data():

    for categ in categories:

         path=os.path.join(data,categ)

         class_num = categories.index(categ)  # get the classification  (0 or a 1). 0=plaque 1=fausse plaque

         for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats

            try:

                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array

                imgnew=cv2.resize(img_array, (150, 150))

                training_data.append([imgnew, class_num])  # add this to our training_data

            except Exception as e:  # in the interest in keeping the output clean...

                pass

        

        

create_training_data()

print(len(training_data ))



import random



random.shuffle(training_data)

 

X = []

y = []



for features,label in training_data:

   X.append(features)

   y.append(label)

  



print (X[0])

print (y)

plt.figure(figsize=(20,20))

for i in range(5):

    plt.subplot(2,5,i+1)

    plt.imshow(X[i])

import seaborn as sns

del training_data

gc.collect()



X=np.array(X)

y=np.array(y)



print('shape x:',X.shape)

print('shape y:',y.shape)



from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.10,random_state=2)

print('shape x_train:',X_train.shape)

print('shape x_val:',X_val.shape)

print('shape x_train:',y_train.shape)

print('shape x_val:',y_val.shape)

 

del X

del y

gc.collect()

n_train=len(X_train)

n_val=len(X_val)



batch_size=32





from keras import layers

from keras import models

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array,load_img

from keras.applications import InceptionResNetV2





conv_base=InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(150,150,3))







model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))



from keras import layers

from keras import models



model=models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))



model.summary()



print('number of trainable weights before freezing the conv base ',len(model.trainable_weights))

conv_base.trainable=False

print('number of trainable weights after freezing the conv base ',len(model.trainable_weights))



model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])



train_datagen=ImageDataGenerator(rescale=1./255,

                                 rotation_range=40,

                                 width_shift_range=0.2,

                                 height_shift_range=0.2,

                                 shear_range=0.2,

                                 zoom_range=0.2,

                                 horizontal_flip=True,)

val_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow(X_train,y_train,batch_size=batch_size)

val_generator=val_datagen.flow(X_val,y_val,batch_size=batch_size)

history=model.fit_generator(train_generator,

                            steps_per_epoch=n_train // batch_size,

                            epochs=64,

                            validation_data=val_generator,

                            validation_steps=n_val // batch_size)





model.save_weights ('model_wieghts.h5') 

model.save ('model_keras.h5')



acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(1,len(acc)+1)



plt.figure()

plt.plot(epochs,acc,'b',label='training accurarcy')

plt.plot(epochs,val_acc,'r',label='Validation accurarcy')

plt.legend(),plt.show()



plt.figure()



plt.plot(epochs,loss,'b',label='Training loss')

plt.plot(epochs,val_loss,'r',label='Validation loss')

plt.title('Training and Validation loss')

plt.legend(),plt.show()





dataa='../input/testtt/test'

test_imgs=['../input/testtt/test/{}'.format(i) for i in os.listdir(dataa)]

X =[] 





for image in test_imgs:

 X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (150,150)))





X= np.array(X)

test_datagen=ImageDataGenerator(rescale=1./255)



i=0

text_labels=[]

plt.figure(figsize=(30,20))

for batch in test_datagen.flow(X,batch_size=1):

    pred=model.predict(batch)

    if pred > 0.5:

         text_labels.append('fausse plaque')

    else:

         text_labels.append(' plaque')

    plt.subplot(2,5,i+1)

    plt.title('This is a '+text_labels[i])

    imgplot=plt.imshow(batch[0])

    i+=1

    if i % 7 == 0:

        break

plt.show()






