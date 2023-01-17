import os
from operator import itemgetter    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

import tensorflow as tf

from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.layers.advanced_activations import PReLU
from keras.datasets import mnist

print(os.getcwd())
print("Modules imported \n")
print("Files in current directory:")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
# LOAD DATA from Kaggle

trainRaw = pd.read_csv('../input/train.csv')
testRaw = pd.read_csv('../input/test.csv')
train = trainRaw.copy()
test_imagesKaggle = testRaw.copy()
train_labelsKaggle = trainRaw['label']

print("train with Labels  ", train.shape)
print("train_labelsKaggle ", train_labelsKaggle.shape)
print("_"*50)
train.drop(['label'],axis=1, inplace=True)
train_imagesKaggle = train
print("train_imagesKaggle without Labels ", train_imagesKaggle.shape)
print("_"*50)
print("test_imagesKaggle  ", test_imagesKaggle.shape)
# RESHAPE to 28 X 28 (Height, Width) which Kaggle has flattened in their file

train4Display = np.array(train_imagesKaggle).reshape(42000,28,28)
test4Display = np.array(test_imagesKaggle).reshape(28000,28,28)

z = 4056

print("train image")
print(train_labelsKaggle[z])
digit = train4Display[z]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

print("test image")
digit = test4Display[z]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# NORMALIZE / SCALE and Prep for CNN in terms of number dimensions expected

train_imagesKaggle = train4Display.reshape(42000,28,28,1)
test_imagesKaggle = test4Display.reshape(28000,28,28,1)

train_imagesKaggle = train_imagesKaggle.astype('float32') / 255
test_imagesKaggle = test_imagesKaggle.astype('float32') / 255
print("train_imagesKaggle ",train_imagesKaggle.shape)
print("test_imagesKaggle ", test_imagesKaggle.shape)
print("_"*50)

# ONE HOT ENCODER for the labels
train_labelsKaggle = to_categorical(train_labelsKaggle)
print("train_labelsKaggle ",train_labelsKaggle.shape)
# Load Data from Keras MNIST

(train_imagesRaw, train_labelsRaw), (test_imagesRaw, test_labelsRaw) = mnist.load_data()
# Normalize / Scale and One Hot encoder for the Keras dataset & Reshape for CNN

train_imagesKeras = train_imagesRaw.copy()
train_labelsKeras = train_labelsRaw.copy()
test_imagesKeras = test_imagesRaw.copy()
test_labelsKeras = test_labelsRaw.copy()

train_imagesKeras = train_imagesKeras.reshape(60000,28,28,1)
test_imagesKeras = test_imagesKeras.reshape(10000,28,28,1)

print("train_imagesKeras ",train_imagesKeras.shape)
print("train_labelsKeras ",train_labelsKeras.shape)
print("test_imagesKeras ", test_imagesKeras.shape)
print("test_labelsKeras ", test_labelsKeras.shape)

# NORMALIZE 0-255 to 0-1
train_imagesKeras = train_imagesKeras.astype('float32') / 255
test_imagesKeras = test_imagesKeras.astype('float32') / 255
print("_"*50)

# ONE HOT ENCODER for the labels
train_labelsKeras = to_categorical(train_labelsKeras)
test_labelsKeras = to_categorical(test_labelsKeras)
print("train_labelsKeras ",train_labelsKeras.shape)
print("test_labelsKeras ", test_labelsKeras.shape)
# CONCATENATE the training sets of Kaggle and Keras into final TRAIN and leave the test for CV

train_images = np.concatenate((train_imagesKeras,train_imagesKaggle), axis=0)
print("new Concatenated train_images ", train_images.shape)
print("_"*50)

train_labels = np.concatenate((train_labelsKeras,train_labelsKaggle), axis=0)
print("new Concatenated train_labels ", train_labels.shape)
# Initial model

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(4, 4), strides=(2,2), input_shape=(28, 28, 1)))
model.add(PReLU())
model.add(layers.Dropout(0.5))
model.add(BatchNormalization())
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (4, 4), strides=(2,2)))
model.add(PReLU())
model.add(layers.Dropout(0.5))
model.add(BatchNormalization())
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), strides=(1,1)))
model.add(PReLU())
model.add(layers.Dropout(0.5))
model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(PReLU())
model.add(layers.Dropout(0.25))
model.add(BatchNormalization())
model.add(layers.Dense(512))
model.add(PReLU())
model.add(layers.Dropout(0.25))
model.add(BatchNormalization())
model.add(layers.Dense(256))
model.add(PReLU())
model.add(layers.Dropout(0.25))
model.add(BatchNormalization())
model.add(layers.Dense(128))
model.add(PReLU())
model.add(BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
# Initial fIT & Evaluate initial model

num_epochs = 80
BatchSize = 2048

model.fit(train_images, train_labels, epochs=num_epochs, batch_size=BatchSize, validation_data=(test_imagesKeras, test_labelsKeras))
test_loss, test_acc = model.evaluate(test_imagesKeras, test_labelsKeras)
print("_"*80)
print("Accuracy on test ", test_acc)
# NN MODEL

def build_model():    
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(4, 4), strides=(2,2), input_shape=(28, 28, 1)))
    model.add(PReLU())
    model.add(layers.Dropout(0.5))
    model.add(BatchNormalization())
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (4, 4), strides=(2,2)))
    model.add(PReLU())
    model.add(layers.Dropout(0.5))
    model.add(BatchNormalization())
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), strides=(1,1)))
    model.add(PReLU())
    model.add(layers.Dropout(0.5))
    model.add(BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(PReLU())
    model.add(layers.Dropout(0.25))
    model.add(BatchNormalization())
    model.add(layers.Dense(512))
    model.add(PReLU())
    model.add(layers.Dropout(0.25))
    model.add(BatchNormalization())
    model.add(layers.Dense(256))
    model.add(PReLU())
    model.add(layers.Dropout(0.25))
    model.add(BatchNormalization())
    model.add(layers.Dense(128))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
# Check some test vs pred
#TestNum = 10
#for t in range(100, 100+TestNum):
#    print(predictions[t])
#    digit = test_imagesRaw[t]
#    plt.imshow(digit, cmap=plt.cm.binary)
#    plt.show()
# CHECK ALL the ERRORS
#TestNum = test_labels.shape[0]
#ErrCount = 0
#for t in range(TestNum):
#        if test_labelsRaw[t] != predictions[t]:
#            ErrCount = ErrCount +1
#            #print("True ", test_labelsRaw[t], "Predicted ",predictions[t])
#            #digit = test_imagesRaw[t]
#            #plt.imshow(digit, cmap=plt.cm.binary)
#            #plt.show()

#print("Errors ", ErrCount, " out of ", TestNum, " = ", 100 * ErrCount/TestNum)
# CROSS VALIDATION k-fold
train_data = train_images
train_targets = train_labels
k = 4
num_val_samples = len(train_data) // k
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
    [train_data[:i * num_val_samples],
    train_data[(i + 1) * num_val_samples:]],
    axis=0)
    partial_train_targets = np.concatenate(
    [train_targets[:i * num_val_samples],
    train_targets[(i + 1) * num_val_samples:]],
    axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
    validation_data=(val_data, val_targets),
    epochs=num_epochs, batch_size=BatchSize, verbose=0)
    
    mae_history = history.history['acc']
    all_mae_histories.append(mae_history)
    
print("Done CV k-fold")
# LOSS Learning curves

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, (len(history.history['acc']) + 1))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# ACCURACY Learning Curves

history_dict = history.history
loss_values = history_dict['acc']
val_loss_values = history_dict['val_acc']
epochs = range(1, (len(history.history['acc']) + 1))
plt.plot(epochs, loss_values, 'bo', label='Training Acc')
plt.plot(epochs, val_loss_values, 'b', label='Validation Acc')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# CONCATENATE the train with test for FINAL FIT

train_imagesFin = np.concatenate((train_images,test_imagesKeras), axis=0)
print("train_imagesFin ", train_imagesFin.shape)
print("_"*50)

train_labelsFin = np.concatenate((train_labels,test_labelsKeras), axis=0)
print("train_labelsFin ", train_labelsFin.shape)
# FINAL FIT according to the above charts

model = build_model()
model.fit(train_imagesFin, train_labelsFin, epochs=num_epochs, batch_size=BatchSize)
# PREDICT & ARGMAX to get the digit from the probability of softmax layer

RawPred = model.predict(test_imagesKaggle)
pred = []
numTest = RawPred.shape[0]
for i in range(numTest):
    pred.append(np.argmax(RawPred[i])) 
predictions = np.array(pred)  
# SUBMISSION
sample_submission = pd.read_csv('../input/sample_submission.csv')
#print(sample_submission.shape)
result=pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':predictions})
result.to_csv("submission.csv",index=False)
print(result)