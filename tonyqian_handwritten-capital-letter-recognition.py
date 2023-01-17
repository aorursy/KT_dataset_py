import numpy as np

import pandas as pd 

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



import keras

from keras import models

from keras.models import Sequential

from keras.layers import Dense,Activation,Conv2D

from keras.layers import MaxPool2D,Flatten,Dropout,ZeroPadding2D,BatchNormalization

from keras.callbacks import ModelCheckpoint

import random

import itertools
#one-hot labeling

def one_hot(labels,num_unique_label=26):

    num_labels=labels.shape[0]

    index_offset=np.arange(num_labels)*num_unique_label

    labels_one_hot=np.zeros((num_labels,num_unique_label))

    labels_one_hot.flat[index_offset+labels.ravel()]=1

    return labels_one_hot   
#Validation function on leave-out test data and showing confusion matrix

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(15, 12))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    #setting axis labels

    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names)

        plt.yticks(tick_marks, target_names)

    #normalize data

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    print('Overall Accuracy=',accuracy,'; Misclassification=',misclass)

    plt.show()
#Reading and preprocessing dataset

alphabet = pd.read_csv('../input/emnist/emnist-letters-train.csv')

#shuffle the data set

#alphabet=alphabet.sample(frac=1)

#split features and labels

images=alphabet.iloc[:,1:].values

raw_labels=alphabet.iloc[:,0].values.ravel()



print('The dimensions of features are',images.shape)

print('The dimensions of raw labels are',raw_labels.shape)
#Show an example of the data

print(raw_labels[15])

tmp=np.transpose(images[15].reshape((28,28)))

plt.imshow(tmp)

print(set(raw_labels))
f,ax = plt.subplots(1,8)

for i in range(8):

    ax[i].imshow(images[i].reshape((28,28)).T,cmap='gray')

plt.show()
labels=one_hot(raw_labels-1)

images=images.reshape(images.shape[0],28,28,1).astype("float32")

images=images/255

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.3)

y = np.array([j for i in Y_test for j in range(len(i)) if i[j] != 0.0])

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'#label for plots in the future
#Creating a simple NN model which serve as baseline

baseline = Sequential()

baseline.add(Flatten())

baseline.add(Dense(512,activation='relu'))

baseline.add(Dropout(0.2))

baseline.add(Dense(26,activation='softmax'))

baseline.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#Fitting baseline model

baseline.fit(x=X_train,y=Y_train,batch_size=300,epochs=50,verbose=1,validation_split=0.2)

baseValLoss = baseline.history.history['val_loss']

baseValAcc = baseline.history.history['val_acc']

baseAcc = baseline.history.history['acc']

baseLoss = baseline.history.history['loss']

epoch = baseline.history.epoch

plt.plot(epoch,baseAcc,'b',label = 'train_acc')

plt.plot(epoch,baseValAcc,'bo',label = 'val_acc')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.title('Baseline Model')

plt.legend()

plt.show()

plt.plot(epoch,baseLoss,'r',label = 'train_loss')

plt.plot(epoch,baseValLoss,'ro',label = 'val_loss')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.title('Baseline Model')

plt.legend()

plt.show()
plt.plot(epoch,baseAcc,'b',label = 'train_acc')

plt.plot(epoch,baseValAcc,'bo',label = 'val_acc')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.title('Baseline Model')

plt.legend()

plt.show()

plt.plot(epoch,baseLoss,'r',label = 'train_loss')

plt.plot(epoch,baseValLoss,'ro',label = 'val_loss')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.title('Baseline Model')

plt.legend()

plt.show()
#Prediction performance of the baseline model

yPredBase = np.array([np.argmax(i) for i in baseline.predict(X_test)])

confusion = confusion_matrix(y, yPredBase, labels=[i for i in range(26)]) 

plot_confusion_matrix(cm=confusion,normalize=False,target_names=[i for i in letters])
# Built the CNN model:

# 3 convolutional steps followed by one fully connected layer for classification

# each convolutional step consists of:

#     1 conv layer with padding to keep to size of input

#     1 conv layer without padding to shrind the size

#     1 pooling layer to reduce size and prevent overfitting

#     1 dropout layer to further prevent overfitting

cnn = Sequential()

#Layer-1

cnn.add(Conv2D(64,kernel_size=(3,3),padding='same',input_shape=(28,28,1),activation='relu'))

cnn.add(Conv2D(64,kernel_size=(3,3),activation='relu'))

cnn.add(MaxPool2D(pool_size=(2,2)))

cnn.add(Dropout(0.5))

#Layer-2

cnn.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))

cnn.add(Conv2D(64,kernel_size=(3,3),activation='relu'))

cnn.add(MaxPool2D(pool_size=(2,2)))

cnn.add(Dropout(0.5))

#Layer-3

cnn.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))

cnn.add(Conv2D(64,kernel_size=(3,3),activation='relu'))

cnn.add(MaxPool2D(pool_size=(2,2)))

cnn.add(Dropout(0.5))

#Fully Connected Layer

cnn.add(Flatten())

cnn.add(Dense(512,activation='relu'))

cnn.add(Dropout(0.2))

cnn.add(Dense(26,activation='softmax'))



cnn.summary()
#Compile the model and set checkpoint to record the best performing parameters

cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

checkpoint=ModelCheckpoint(filepath="best_weights.hdf5",monitor='val_acc',save_best_only=True)
#Fit the model with 50 full runs

cnn.fit(x=images,y=labels,batch_size=500,epochs=50,verbose=1,

          callbacks=[checkpoint],validation_split=0.2)
#Model performance

cnnValLoss = cnn.history.history['val_loss']

cnnValAcc = cnn.history.history['val_acc']

cnnAcc = cnn.history.history['acc']

cnnLoss = cnn.history.history['loss']

epoch = cnn.history.epoch

plt.plot(epoch,cnnAcc,'b',label = 'train_acc')

plt.plot(epoch,cnnValAcc,'bo',label = 'val_acc')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.title('CNN with 3 Conv Layers')

plt.legend()

plt.figure()

plt.plot(epoch,cnnLoss,'r',label = 'train_loss')

plt.plot(epoch,cnnValLoss,'ro',label = 'val_loss')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.title('CNN with 3 Conv Layers')

plt.legend()

plt.show()
#Load the saved optimal weights as the final model

cnn.load_weights('best_weights.hdf5')

cnn.save('shapes_cnn.h5')
#Model performance on leave-out test data

yPredCNN = np.array([np.argmax(i) for i in cnn.predict(X_test)])

confusionCNN = confusion_matrix(y, yPredCNN, labels=[i for i in range(26)]) 

plot_confusion_matrix(cm=confusionCNN,normalize=False,target_names=[i for i in letters])
example = pd.read_csv('../input/emnisttest/emnist-letters-test.csv',nrows=1)

imgEx = example.iloc[:,1:].values

labelEx = example.iloc[:,0].values.ravel()

print('The given label of this example is: ',labelEx[0],letters[labelEx[0]-1])

plt.imshow(np.transpose(imgEx[0].reshape((28,28))))

plt.show()

imgEX = imgEx.reshape(imgEx.shape[0],28,28,1).astype("float32")
imgEx=imgEx.reshape(imgEx.shape[0],28,28,1).astype("float32")

print(imgEx.shape)
# predicting images

imgExPred = np.argmax(cnn.predict(imgEx))

print("Predicted class is: ",imgExPred,letters[imgExPred])
# Extracts the outputs of the top 12 layers

layer_outputs = [layer.output for layer in cnn.layers[:12]] 

# Creates a model that will return these outputs, given the model input

activation_cnn = models.Model(inputs=cnn.input, outputs=layer_outputs)

# Returns a list of five Numpy arrays: one array per layer activation

activations = activation_cnn.predict(imgEx) 



layer1Activation = activations[0]

print(layer1Activation.shape)

plt.matshow(np.transpose(layer1Activation[0, :, :, 4]), cmap='gray')
layer_names = []

for layer in cnn.layers[:12]:

    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    

images_per_row = 16



for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps

    n_features = layer_activation.shape[-1] # Number of features in the feature map

    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).

    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols): # Tiles each filter into a big horizontal grid

        for row in range(images_per_row):

            channel_image = layer_activation[0,

                                             :, :,

                                             col * images_per_row + row]

            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size, # Displays the grid

                         row * size : (row + 1) * size] = np.transpose(channel_image)

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='gray')
rawOut = cnn.predict(imgEx)

print(rawOut)