

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sb



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
Y_train = train["label"]

# Dropeamos la columna 'label' del set de entrenamiento 

X_train = train.drop(labels = ["label"],axis = 1)



# liberamos espacio

del train 



#Graficamos la cantidad de veces que se repite un número.

g = sb.countplot(Y_train)

Y_train.value_counts()



print(X_train.shape)

print(test.shape)
print(X_train.isnull().any().describe())

print("___________________________")

print(test.isnull().any().describe())

X_train = X_train / 255.0

test = test / 255.0



#Reshape de las imágenes, utilizando "channel last" 

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



#One hot encoding

Y_train = to_categorical(Y_train, num_classes = 10)



#Dividimos el set de entrenamiento



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 1)



# [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()





#Conv - Conv - Maxpool2D

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



#Conv - Conv - Maxpool2D

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))



model.add(Dense(10, activation = "softmax"))
#Definimos el optimizador

optimizer = Adam()  #lr=0.001



#Compilamos

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

datagen = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False, 

        rotation_range=10,  #rota la imagen de 0 a 180 grados

        zoom_range = 0.1, #Hace zoom a la imágen 

        width_shift_range=0.1,  # randomly shift images horizontally 

        height_shift_range=0.1,  # randomly shift images vertically

        horizontal_flip=False,  

        vertical_flip=False)  



datagen.fit(X_train)
epochs = 30 

batch_size = 86



history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size)

predictions = model.predict_classes(test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("vvcp2.csv", index=False, header=True)