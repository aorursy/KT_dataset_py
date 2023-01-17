import pandas as pd

import numpy as np

from matplotlib import pyplot as plt



from keras.models import Sequential, load_model



from keras.layers import Dense, Dropout, Flatten, Activation



from keras.layers.convolutional import Conv2D, MaxPooling2D



from keras.utils.np_utils import to_categorical



import seaborn as sns



from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
from keras.datasets import cifar10



(X_train, y_train), (X_test, y_test) = cifar10.load_data()
cifar_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

y_train=to_categorical(y_train)

y_test=to_categorical(y_test)

num_classes=y_train[0].shape[0]
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



valid_datagen = ImageDataGenerator(

        rescale=1./255)
train_generator = train_datagen.flow(X_train,y_train)

valid_generator = valid_datagen.flow(X_test,y_test)
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=(32,32,3)))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
STEP_SIZE_TRAIN=5*train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=10

)
from sklearn.metrics import accuracy_score as acc

y_pred=model.predict(X_test)

y_pred_classe=y_pred.argmax(axis=-1)

y_test_classe=y_test.argmax(axis=-1)



print(acc(y_test_classe,y_pred_classe))



cm = confusion_matrix(y_test_classe,y_pred_classe)

plt.figure(figsize = (12,10))

sns.heatmap(cm, annot=True, cmap="coolwarm")
import random

plt.figure(figsize=(15,25))

n_test = X_test.shape[0]

for i in range(1,50) :

    ir = random.randint(0,n_test)

    plt.subplot(10,5,i)

    plt.axis('off')

    plt.imshow(X_test[ir])

    pred_classe = y_pred_classe[ir]

    plt.title(cifar_classes[pred_classe])