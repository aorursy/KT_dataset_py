import os,cv2

!pip install keract

import numpy as np

import matplotlib.pyplot as plt



from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from numpy import *



from keras import backend as K

K.set_image_dim_ordering('th')



from keras.utils import np_utils

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.optimizers import SGD,RMSprop,adam

from keras.constraints import maxnorm

from PIL import Image

from PIL import ImageMath

files_trainNO = '../input/cellphone/training/training/cellphone-NO/'

files_trainYES = '../input/cellphone/training/training/cellphone-YES/'

files_OS_trainNO = os.listdir(files_trainNO)

files_OS_trainYES = os.listdir(files_trainYES)

trainingSamples = size(files_OS_trainNO) + size(files_OS_trainYES)

print(' No cellphone samples:', size(files_OS_trainNO))

print(' Yes cellphone samples:', size(files_OS_trainYES))

print(' Total training samples:', trainingSamples)
immatrixNo = array([array(Image.open('{}img{}.png'.format(files_trainNO, image))).flatten()

                          for image in range(size(files_OS_trainNO))], 'f')



immatrixYes = array([array(Image.open('{}img{}.png'.format(files_trainYES, 121 + image))).flatten()

                          for image in range(size(files_OS_trainYES))], 'f')

    

immatrix = np.concatenate((immatrixNo, immatrixYes), axis=0)

labelSamples = np.ones((trainingSamples,), dtype=int)

labelSamples[0:120] = 0

labelSamples[120:] = 1



cellphone = immatrixYes[123]



nb_epoch = 30

rows, cols = 90, 90

n_channels = 1

batch_size = 32

n_classes = 2

n_filter = 32

n_pool = 2

n_conv = 3



print("If value = 0 , then : THERE IS NOT A CELLPHONE !")

print("If value = 1 , then : THERE IS A CELLPHONE !")



data, label = shuffle(immatrix, labelSamples, random_state=2)

train_data = [data, label]

print(train_data)
def show_Samples(show, j):

    s = [0, 30, 60, 130, 200, 190]

    fig, axs = plt.subplots(1, show, figsize=(10, 3))

    for i in range(0, show):

        anyOne = random.randint(0, 250)

        axs[i].imshow(immatrix[s[i] + j].reshape(rows, cols))

        



for i in range(0, 6):

    show_Samples(6, i)

(X,y) = (train_data[0],train_data[1])

X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=4)

cellphone = X_train[125]

X_train = X_train.reshape(X_train.shape[0],1,rows,cols)

cellphone = cellphone.reshape(1, 1, 90, 90)



X_test = X_test.reshape(X_test.shape[0],1,rows,cols)

X_train = X_train.astype("float32")

X_test = X_test.astype("float32")

X_train /= 255

X_test /= 255



Y_train = np_utils.to_categorical(y_train,n_classes)

Y_test = np_utils.to_categorical(y_test,n_classes)
model = Sequential()



model.add(Conv2D(n_filter,(n_conv,n_conv),border_mode='same',input_shape=(n_channels,rows,cols)))

model.add(Activation('relu'))

model.add(Conv2D(n_filter,(n_conv,n_conv)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(n_pool,n_pool)))

model.add(Dropout(0.5))



model.add(Conv2D(64,(n_conv,n_conv)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(n_pool,n_pool)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(128))

model.add(Dropout(0.5))

model.add(Dense(n_classes))

model.add(Activation('softmax'));



model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
from keract import get_activations

activations = get_activations(model, cellphone)



plt.imshow(cellphone.reshape(90, 90))
from keract import display_activations

display_activations(activations, save=False)
from keract import display_heatmaps

display_heatmaps(activations, cellphone.reshape(1, 90, 90, 1), save=False)
history = model.fit(X_train,Y_train,batch_size = batch_size, epochs = nb_epoch, verbose=1)
def show_loss():

    plt.figure(figsize=(10,3))

    plt.plot(history.history['loss'], 

             color='red', 

             label='Loss')

    plt.title("My Test Loss")

    plt.xlabel("Epochs")

    plt.ylabel("Loss")

    plt.legend()

    plt.show()

    

def show_acc_epochs():

    plt.figure(figsize=(10,3))

    plt.plot(history.history['acc'],

             color='blue',

             label='Acc')

    plt.title("My Test Accuracy")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy Value")

    plt.legend()

    plt.show()

    

    

show_loss()

show_acc_epochs()
answer = model.predict_classes(X_test)



test_pred = X_test

def show_Samples_test(show, test_pred, answer):

    fig, axs = plt.subplots(1, show, figsize=(10, 3))

    for i in range(0, show):

        axs[i].imshow(test_pred[i].reshape(rows, cols), label=answer[i])

        axs[i].set_title('Label: {}'.format(answer[i]))

        leg = axs[i].legend()

        

show_Samples_test(6, test_pred, answer)