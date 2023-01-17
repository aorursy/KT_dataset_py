# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input, ZeroPadding2D
from keras.models import Sequential
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import h5py

def load_dataset():
    train_data = h5py.File('../input/happy-house-dataset/train_happy.h5', 'r')
    X_train = np.array(train_data["train_set_x"][:450])
    Y_train = np.array(train_data["train_set_y"][:450])
    
    X_val = np.array(train_data["train_set_x"][:150])
    Y_val = np.array(train_data["train_set_y"][:150])
    
    test_data = h5py.File('../input/happy-house-dataset/test_happy.h5', 'r')
    X_test = np.array(test_data["test_set_x"][:])
    Y_test = np.array(test_data["test_set_y"][:])
    
    Y_train = Y_train.reshape((Y_train.shape[0],1))
    Y_val = Y_val.reshape((Y_val.shape[0],1))
    Y_test = Y_test.reshape((Y_test.shape[0],1))
    
    return (X_train,Y_train), (X_val, Y_val), (X_test,Y_test)
(X_train,Y_train), (X_val, Y_val), (X_test,Y_test) = load_dataset()
print("Training Set -: X shape - {}, Y shape - {}".format(X_train.shape, Y_train.shape))
print("Validation (Dev - test) Set -: X shape - {}, Y shape - {}".format(X_val.shape, Y_val.shape))
print("Test Set -: X shape - {}, Y shape - {}".format(X_test.shape, Y_test.shape))
#normalize the image matix -> vectors 
X_train = X_train/255
X_test = X_test/255
X_val = X_val/255
print (X_train.shape[0])
print (X_val.shape[0])
print (X_test.shape[0])

#padding = 0 => 'valid/same convolution'
#filter/kernal = 5, 5,  
#Dropout - Inputs randomly eliminates,features away randomly, weight away randomly
def buildModel(): 
    model = Sequential()
    model.add(Conv2D(32, 5, activation= 'relu', strides=(1, 1), input_shape=(64, 64, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Conv2D(16, 5, activation= 'relu', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()
    return model;      
test = buildModel()
plot_model(test, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
test.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard , CSVLogger, ReduceLROnPlateau
checkpoint_sdg = ModelCheckpoint("model_sdg.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
checkpoint_rms = ModelCheckpoint("model_rms.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
checkpoint_adam = ModelCheckpoint("model_adam.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
checkpoint_l5 = ModelCheckpoint("model_l5.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
from keras.optimizers import SGD
#With validation data, compare against validation data (dev - test)
#If steps_per_epoch is set, the `batch_size` must be None.
#batch size = 2**x , 16,32,64,24 
epoch = 25
steps_per_epoch = 25
learning_rate = 0.01
validation_steps = 15
batch_size = 32
model_sgd = buildModel()
optimizer = SGD(lr=learning_rate, nesterov=True)
model_sgd.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
history = model_sgd.fit(X_train,Y_train,epochs=epoch,batch_size=None, steps_per_epoch = steps_per_epoch,validation_data=(X_val, Y_val), validation_steps=validation_steps,callbacks=[checkpoint_sdg])
scores = model_sgd.evaluate(X_test, Y_test, verbose=0)
loss_valid=scores[0]
acc_valid=scores[1]

y_pred = model_sgd.predict_classes(X_test)
test_accuracy = accuracy_score(y_pred, Y_test)

print("Test Accuracy:{:.01%}".format(test_accuracy))
print('Recall Score:', recall_score(y_pred, Y_test))
print('Precision Score:', precision_score(y_pred, Y_test))
print('F1 Score:', f1_score(y_pred, Y_test))

print('-------------------SGD-----------------------------------------')
print("validation loss: {:.2f}, validation accuracy: {:.01%}".
              format(loss_valid, acc_valid))
print('---------------------------------------------------------------')
import seaborn as sns
import matplotlib.pyplot as plt
# confusion matrix
import seaborn as sns
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, y_pred) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="BuPu",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


import matplotlib.pyplot as plt

acc = history.history['accuracy']
loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize = (16, 5))

plt.subplot(1,2,1)
plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

from keras.optimizers import RMSprop
#With validation data, compare against validation data (dev - test)
#If steps_per_epoch is set, the `batch_size` must be None.
#batch size = 2**x , 16,32,64,24 
# epoch = 5
# learning_rate = 0.01
# validation_steps = 15
# batch_size = 32
model_rmsprop = buildModel()
optimizer = RMSprop(lr=learning_rate)
model_rmsprop.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
history = model_rmsprop.fit(X_train,Y_train,epochs=epoch,batch_size=None, steps_per_epoch = steps_per_epoch,validation_data=(X_val, Y_val), validation_steps=validation_steps,callbacks=[checkpoint_rms])
scores = model_rmsprop.evaluate(X_test, Y_test, verbose=1)
loss_valid=scores[0]
acc_valid=scores[1]

y_pred = model_rmsprop.predict_classes(X_test)
test_accuracy = accuracy_score(y_pred, Y_test)

print("Test Accuracy:{:.01%}".format(test_accuracy))
print('Recall Score:', recall_score(y_pred, Y_test))
print('Precision Score:', precision_score(y_pred, Y_test))
print('F1 Score:', f1_score(y_pred, Y_test))

print('-------------------RMSprop-----------------------------------------')
print("validation loss: {:.2f}, validation accuracy: {:.01%}".
              format(loss_valid, acc_valid))
print('-------------------------------------------------------------------')
import seaborn as sns
import matplotlib.pyplot as plt
# confusion matrix
import seaborn as sns
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, y_pred) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="BuPu",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
import matplotlib.pyplot as plt

acc = history.history['accuracy']
loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize = (16, 5))

plt.subplot(1,2,1)
plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

from keras.optimizers import Adam
#With validation data, compare against validation data (dev - test)
#If steps_per_epoch is set, the `batch_size` must be None.
#batch size = 2**x , 16,32,64,24 
# epoch = 25
# learning_rate = 0.01
# validation_steps = 15
# batch_size = 32
# steps_per_epoch = 4
model_adam = buildModel()
optimizer = Adam(lr=learning_rate)
model_adam.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
history = model_adam.fit(X_train,Y_train,epochs=epoch,batch_size=None, steps_per_epoch = steps_per_epoch,validation_data=(X_val, Y_val), validation_steps=validation_steps,callbacks=[checkpoint_adam])
scores = model_adam.evaluate(X_test, Y_test, verbose=0)
loss_valid=scores[0]
acc_valid=scores[1]

y_pred = model_adam.predict_classes(X_test)
test_accuracy = accuracy_score(y_pred, Y_test)

print("Test Accuracy:{:.01%}".format(test_accuracy))
print('Recall Score:', recall_score(y_pred, Y_test))
print('Precision Score:', precision_score(y_pred, Y_test))
print('F1 Score:', f1_score(y_pred, Y_test))

print('-------------------ADAM----------------------------------------')
print("validation loss: {:.2f}, validation accuracy: {:.01%}".
              format(loss_valid, acc_valid))
print('-------------------------------------------------------------------')
import seaborn as sns
import matplotlib.pyplot as plt
# confusion matrix
import seaborn as sns
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, y_pred) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="BuPu",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
import matplotlib.pyplot as plt

acc = history.history['accuracy']
loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize = (16, 5))

plt.subplot(1,2,1)
plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
#1X1 conv used to reduce number of parameters
def buildLModel(): 
    model = Sequential()
    model.add(Conv2D(1, 1, activation= 'relu', strides=(2, 2), input_shape=(64, 64, 3)))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(2))
    model.add(Conv2D(6, 5, activation= 'relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(2))
    model.add(Conv2D(16, 5, activation= 'relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model;     
modelLet = buildLModel()
plot_model(modelLet, to_file='model_let_plot.png', show_shapes=True, show_layer_names=True)
modelLet.summary()
from keras.optimizers import Adam
#With validation data, compare against validation data (dev - test)
#If steps_per_epoch is set, the `batch_size` must be None.
#batch size = 2**x , 16,32,64,24 
# epoch = 25
# learning_rate = 0.01
# validation_steps = 15
# batch_size = 32
# steps_per_epoch = 24
model_adam = buildLModel()
optimizer = Adam(lr=learning_rate)
model_adam.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
history = model_adam.fit(X_train,Y_train,epochs=epoch,batch_size=None,steps_per_epoch = steps_per_epoch,validation_data=(X_val, Y_val), validation_steps=validation_steps,callbacks=[checkpoint_l5])
scores = model_adam.evaluate(X_test, Y_test, verbose=0)
loss_valid=scores[0]
acc_valid=scores[1]

y_pred = model_adam.predict_classes(X_test)
test_accuracy = accuracy_score(y_pred, Y_test)

print("Test Accuracy:{:.01%}".format(test_accuracy))
print('Recall Score:', recall_score(y_pred, Y_test))
print('Precision Score:', precision_score(y_pred, Y_test))
print('F1 Score:', f1_score(y_pred, Y_test))

print('-------------------ADAM-----------------------------------------')
print("validation loss: {:.2f}, validation accuracy: {:.01%}".
              format(loss_valid, acc_valid))
print('-------------------------------------------------------------------')
import seaborn as sns
import matplotlib.pyplot as plt
# confusion matrix
import seaborn as sns
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, y_pred) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="BuPu",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
import matplotlib.pyplot as plt

acc = history.history['accuracy']
loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize = (16, 5))

plt.subplot(1,2,1)
plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()