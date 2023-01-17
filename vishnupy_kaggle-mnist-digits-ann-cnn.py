import pandas as pd

import numpy as np



np.random.seed(1212)



import keras

from keras.models import Model

from keras.layers import *

from keras.layers import Dense

from keras import optimizers
!pwd

!ls
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')

#images in these datasets are already flattened and in Machine readable format
df_train.head() # 784 features, 1 label
df_features = df_train.iloc[:, 1:785]

df_label = df_train.iloc[:, 0]



X_test = df_test.iloc[:, 0:784]



print(X_test.shape)
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(df_features, df_label, test_size = 0.2, random_state = 42)

X_train = X_train.values.reshape(33600, 784) #(33600, 784)

X_cv = X_cv.values.reshape(8400, 784) #(8400, 784)

X_test = X_test.as_matrix().reshape(28000, 784)
X_train.shape, X_cv.shape, y_train.shape, X_test.shape
print((min(X_train[1]), max(X_train[1])))
# Feature Normalization 

X_train = X_train.astype('float32'); 

X_cv    = X_cv.astype('float32'); 

X_test  = X_test.astype('float32')

X_train /= 255; 

X_cv    /= 255; 

X_test  /= 255



# Convert labels to One Hot Encoded

num_digits = 10

y_train = keras.utils.to_categorical(y_train, num_digits)

y_cv = keras.utils.to_categorical(y_cv, num_digits)
X_train.shape, X_cv.shape, y_train.shape, X_test.shape, y_cv.shape
# Printing 2 examples of labels after conversion

print(y_train[0]) # 2

print(y_train[3]) # 7
# Input Parameters

n_input = 784 # number of features

n_hidden_1 = 300

n_hidden_2 = 100

n_hidden_3 = 100

n_hidden_4 = 200

num_digits = 10
Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)
# Our model would have '6' layers - input layer, 4 hidden layer and 1 output layer

model = Model(Inp, output)

model.summary() # We have 297,910 parameters to estimate
# Insert Hyperparameters

learning_rate = 0.1

training_epochs = 20

batch_size = 100

sgd = optimizers.SGD(lr=learning_rate)
# We rely on the plain vanilla Stochastic Gradient Descent as our optimizing methodology

model.compile(loss='categorical_crossentropy',

              optimizer='sgd',

              metrics=['accuracy'])
history1 = model.fit(X_train, y_train,

                     batch_size = batch_size,

                     epochs = training_epochs,

                     verbose = 2,

                     validation_data=(X_cv, y_cv))
_, val_acc1 = model.evaluate(X_cv, y_cv)

print (val_acc1)
Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)



# We rely on ADAM as our optimizing methodology

adam = keras.optimizers.Adam(lr=learning_rate)

model2 = Model(Inp, output)



model2.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history2 = model2.fit(X_train, y_train,

                      batch_size = batch_size,

                      epochs = training_epochs,

                      verbose = 2,

                      validation_data=(X_cv, y_cv))
_, val_acc2 = model2.evaluate(X_cv, y_cv)

print (val_acc2)
Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)



learning_rate = 0.01

adam = keras.optimizers.Adam(lr=learning_rate)

model2a = Model(Inp, output)



model2a.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history2a = model2a.fit(X_train, y_train,

                        batch_size = batch_size,

                        epochs = training_epochs,

                        verbose = 2,

                        validation_data=(X_cv, y_cv))
_, val_acc2a = model2a.evaluate(X_cv, y_cv)

print (val_acc2a)
Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)



learning_rate = 0.5

adam = keras.optimizers.Adam(lr=learning_rate)

model2b = Model(Inp, output)



model2b.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history2b = model2b.fit(X_train, y_train,

                        batch_size = batch_size,

                        epochs = training_epochs,

                            validation_data=(X_cv, y_cv))
_, val_acc2b = model2b.evaluate(X_cv, y_cv)

print (val_acc2b)
# Input Parameters

n_input = 784 # number of features

n_hidden_1 = 300

n_hidden_2 = 100

n_hidden_3 = 100

n_hidden_4 = 100

n_hidden_5 = 200

num_digits = 10
Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

x = Dense(n_hidden_5, activation='relu', name = "Hidden_Layer_5")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)
# Our model would have '7' layers - input layer, 5 hidden layer and 1 output layer

model3 = Model(Inp, output)

model3.summary() # We have 308,010 parameters to estimate
# We rely on 'Adam' as our optimizing methodology

adam = keras.optimizers.Adam(lr=0.01)



model3.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



history3 = model3.fit(X_train, y_train,

                      batch_size = batch_size,

                      epochs = training_epochs,

                      validation_data=(X_cv, y_cv))
_, val_acc3 = model3.evaluate(X_cv, y_cv)

print (val_acc3)
# Input Parameters

n_input = 784 # number of features

n_hidden_1 = 300

n_hidden_2 = 100

n_hidden_3 = 100

n_hidden_4 = 200

num_digits = 10
Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dropout(0.3)(x)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dropout(0.3)(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dropout(0.3)(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)
# Our model would have '6' layers - input layer, 4 hidden layer and 1 output layer

model4 = Model(Inp, output)

model4.summary() # We have 297,910 parameters to estimate
model4.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model4.fit(X_train, y_train,

                    batch_size = batch_size,

                    epochs = training_epochs,

                    validation_data=(X_cv, y_cv))
_, val_acc4 = model4.evaluate(X_cv, y_cv)

print (val_acc4)
eval_model = pd.DataFrame({

"Model": ['Model', 'Model2', 'Model2a', 'Model2b', 'Model3', 'Model4'],

    "Validation Accuracy" : [val_acc1, val_acc2, val_acc2a, val_acc2b, val_acc3, val_acc4]

   })



eval_model.sort_values(by = 'Validation Accuracy', ascending=False)
test_pred = pd.DataFrame(model4.predict(X_test, batch_size=200))

test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))

test_pred.index.name = 'ImageId'

test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()

test_pred['ImageId'] = test_pred['ImageId'] + 1



test_pred.head()
#test_pred.to_csv('kaggle_mnist_digits_submission.csv', index = False)
# Import MNIST dataset which is already in Keras



from tensorflow.keras.datasets import fashion_mnist



# Import the Libraries



#Same as ANN

from keras.models import Sequential



# Related to CNN

from keras.layers import Conv2D

from keras.layers import MaxPool2D

from keras.layers import Flatten

from tensorflow.keras.utils import to_categorical



import matplotlib.pyplot as plt

%matplotlib inline



#Same ANN Model

from keras.layers import Dense



# Get data

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

"""

import os

from six.moves.urllib.request import urlretrieve

import sys

import gzip

import _pickle as cPickle

import pickle



f = gzip.open('../input/mnist-data/mnist.pkl/mnist.pkl', 'rb')

data = cPickle.load(f, encoding='bytes')

f.close()

#(X_train, y_train), (X_test, y_test) = data

(X_train, _), (X_test, _) = data

"""

print('Training data shape : ', X_train.shape, y_train.shape)

print ('Test Shape ',  X_test.shape, y_test.shape)
# Find the unique numbers from the train labels

classes = np.unique(y_train)

nClasses = len(classes)

print('Total number of outputs : ', nClasses)

print('Output classes : ', classes)
plt.figure(figsize=[5, 5])



#Display first image in training data

plt.subplot(121)

plt.imshow(X_train[0,:,:], cmap='gray')

plt.title("Ground Truth : {}".format(y_train[0]))



# Display the first image in testing data

plt.subplot(122)

plt.imshow(X_test[0,:,:], cmap='gray')

plt.title("Ground Truth : {}".format(y_test[0]))
X_train = X_train.reshape(-1, 28,28, 1)

X_test = X_test.reshape(-1, 28,28, 1)

print ('Training datasets shapes', X_train.shape, X_test.shape)



X_train.dtype
# Change the labels from categorical to one-hot encoding

y_train_one_hot = to_categorical(y_train)

y_test_one_hot = to_categorical(y_test)



# Display the change for category label using one-hot encoding

print('Original label:', y_train[0])

print('After conversion to one-hot:', y_train_one_hot[0])
from sklearn.model_selection import train_test_split

X_train,X_valid,label_train,label_valid = train_test_split(X_train, y_train_one_hot, test_size=0.2, random_state=13)
# Check the Shapes of the Train and  Validation Datasets

X_train.shape,X_valid.shape,label_train.shape, label_valid.shape
# these were not included earlier but now will add these to use for better performance

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

batch_size = 64

epochs = 20

num_classes = 10
fashion_model = Sequential()

fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(MaxPooling2D((2, 2),padding='same'))

fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))                  

fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

fashion_model.add(Flatten())

fashion_model.add(Dense(128, activation='linear'))

fashion_model.add(LeakyReLU(alpha=0.1))                  

fashion_model.add(Dense(num_classes, activation='softmax'))
fashion_model.summary()
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
batch_size = 64

epochs = 20

num_classes = 10
fashion_train = fashion_model.fit(X_train, label_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_valid, label_valid))
test_eval = fashion_model.evaluate(X_test, y_test_one_hot, verbose=0)



print('Test loss:', test_eval[0])

print('Test accuracy:', test_eval[1])
accuracy = fashion_train.history['acc']

val_accuracy = fashion_train.history['val_acc']

loss = fashion_train.history['loss']

val_loss = fashion_train.history['val_loss']

epochs = range(len(accuracy))

plt.figure(figsize=[10,5])

plt.subplot(121)

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy', color='red' )

plt.title('Training and validation accuracy')

plt.legend()



plt.subplot(122)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss', color='red' )

plt.title('Training and validation loss')

plt.legend()

plt.show()
fashion_model = Sequential()

fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(MaxPooling2D((2, 2),padding='same'))

fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))                  

fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

fashion_model.add(Dropout(0.4))

fashion_model.add(Flatten())

fashion_model.add(Dense(128, activation='linear'))

fashion_model.add(LeakyReLU(alpha=0.1))           

fashion_model.add(Dropout(0.3))

fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.summary()
batch_size = 64

epochs = 20

num_classes = 10
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])



fashion_train_dropout = fashion_model.fit(X_train, label_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_valid, label_valid))
test_eval = fashion_model.evaluate(X_test, y_test_one_hot, verbose=1)

print('Test loss:', test_eval[0])

print('Test accuracy:', test_eval[1])
accuracy = fashion_train_dropout.history['acc']

val_accuracy = fashion_train_dropout.history['val_acc']

loss = fashion_train_dropout.history['loss']

val_loss = fashion_train_dropout.history['val_loss']

epochs = range(len(accuracy))

plt.figure(figsize=[10,5])

plt.subplot(121)

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy', color='red' )

plt.title('Training and validation accuracy')

plt.legend()



plt.subplot(122)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss', color='red' )

plt.title('Training and validation loss')

plt.legend()

plt.show()
predicted_classes = fashion_model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, y_test.shape
correct = np.where(predicted_classes==y_test)[0]

print ("Found %d correct labels" % len(correct))

for i, correct in enumerate(correct[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

    plt.tight_layout()
incorrect = np.where(predicted_classes!=y_test)[0]

print ("Found %d incorrect labels" % len(incorrect))

for i, incorrect in enumerate(incorrect[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

    plt.tight_layout()
from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_test, predicted_classes, target_names=target_names))