#python basics

from matplotlib import pyplot as plt

import math, os, re, time, random

import numpy as np, pandas as pd, seaborn as sns



#bare minimum to use tensorflow

import tensorflow as tf



#for model evaluation

from sklearn.model_selection import train_test_split
#create a rank 0 tensor

rank_0_tensor = tf.constant(1)

print(rank_0_tensor); print('')



#create a rank 1 tensor

rank_1_tensor = tf.constant([1, 0, 0])

print(rank_1_tensor); print('')



#create a rank 2 tensor

rank_2_tensor = tf.constant([[1, 0, 0],

                             [0, 1, 0],

                             [0, 0, 1]])

print(rank_2_tensor)
#create a rank 0 tensor

rank_0_tensor = tf.constant(1, dtype = tf.float16)

print(rank_0_tensor); print('')



#create a rank 1 tensor

rank_1_tensor = tf.constant([1, 0, 0], dtype = tf.float32)

print(rank_1_tensor); print('')



#create a rank 2 tensor

rank_2_tensor = tf.constant([[1, 0, 0],

                             [0, 1, 0],

                             [0, 0, 1]], dtype = tf.int32)

print(rank_2_tensor)
print(f"Element type: {rank_2_tensor.dtype}")

print(f"Number of dimensions: {rank_2_tensor.ndim}")

print(f"Shape of tensor: {rank_2_tensor.shape}")

print(f"Elements along axis 0 of tensor: {rank_2_tensor.shape[0]}")

print(f"Elements along the last axis of tensor: {rank_2_tensor.shape[-1]}")

print(f"Total number of elements: {tf.size(rank_2_tensor).numpy()}")
#convert to NumPy array with .numpy()

print(type(rank_2_tensor.numpy()))

print(rank_2_tensor.numpy()); print('')



#convert to NumPy array by performing np operation on it

tensor_to_array = np.add(rank_2_tensor, 1)

print(type(tensor_to_array))

print(tensor_to_array); print('')



#convert to tensor by performing tf operation on it

array_to_tensor = tf.add(rank_2_tensor.numpy(), 1)

print(array_to_tensor)
model = tf.keras.models.Sequential()



#add relu activated layer with 256 nodes

model.add(tf.keras.layers.Dense(256, activation='relu', input_shape = (784,)))



#add swish activated layer with 128 nodes

model.add(tf.keras.layers.Dense(128, activation='swish'))



#add softmax output layer that predicts 10 different classes

model.add(tf.keras.layers.Dense(10, activation='softmax'))



#compile model 

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])



#check model architecture

model.summary()
model = tf.keras.models.Sequential([



#add relu activated layer with 256 nodes

tf.keras.layers.Dense(256, activation = 'relu', input_shape = (784,)),



#add swish activated layer with 128 nodes

tf.keras.layers.Dense(128, activation = 'swish'),



#add softmax output layer that predicts 10 different classes

tf.keras.layers.Dense(10, activation = 'softmax')

    

])



#compile model 

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["categorical_accuracy"])



#check model architecture

model.summary()
#define an input

inputs = tf.keras.Input(shape = (784,))



#add a relu layer to act on the inputs

x = tf.keras.layers.Dense(256, activation = 'relu')(inputs)



#repeat with swish layer

x = tf.keras.layers.Dense(128, activation = 'swish')(x)



#define outputs

outputs = tf.keras.layers.Dense(10, activation = 'softmax')(x)



#build model with inputs/outputs

model = tf.keras.Model(inputs = inputs, outputs = outputs)



#compile model 

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["categorical_accuracy"])



#check model architecture

model.summary()
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')



#sneak peek

train.head()
#sneak peek

test.head()
#remove label from train and store as separate dataframe

labels = train['label']

train = train.drop('label', axis = 1)



train = train / 255.0

test = test / 255.0
train = train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
labels = tf.one_hot(labels, depth = 10).numpy()
#view sample image from our training set

plt.imshow(train[5][:,:,0], cmap = plt.cm.binary);
#demo ImageDataGenerator functionality

datagen = tf.keras.preprocessing.image.ImageDataGenerator(

                                                          rotation_range = 20,  

                                                          zoom_range = 0.1,  

                                                          width_shift_range = 0.1, 

                                                          height_shift_range = 0.1

)
#preview augmented images from generator

sample = train[9,].reshape((1,28,28,1))

sample_labels = labels[9,].reshape((1,10))

plt.figure(figsize=(15,4.5))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    train2, labels2 = datagen.flow(sample,sample_labels).next()

    plt.imshow(train2[0].reshape((28,28)),cmap = plt.cm.binary)

    plt.axis('off')

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
#how many epochs to train each CNN

EPOCHS = 45



#batch size of each CNN

BATCH_SIZE = 64



#how many CNNs to create and train

NUM_NETS = 25



#for training progress

VERBOSE = 0
#build and train NUM_NETS different CNNs

model = [0] * NUM_NETS

for j in range(NUM_NETS):

    model[j] = tf.keras.models.Sequential()



    model[j].add(tf.keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu', input_shape = (28, 28, 1)))

    model[j].add(tf.keras.layers.BatchNormalization())

    model[j].add(tf.keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu'))

    model[j].add(tf.keras.layers.BatchNormalization())

    model[j].add(tf.keras.layers.Conv2D(32, kernel_size = 5, strides = 2, padding = 'same', activation = 'relu'))

    model[j].add(tf.keras.layers.BatchNormalization())

    model[j].add(tf.keras.layers.Dropout(0.4))



    model[j].add(tf.keras.layers.Conv2D(64, kernel_size = 3, activation = 'relu'))

    model[j].add(tf.keras.layers.BatchNormalization())

    model[j].add(tf.keras.layers.Conv2D(64, kernel_size = 3, activation = 'relu'))

    model[j].add(tf.keras.layers.BatchNormalization())

    model[j].add(tf.keras.layers.Conv2D(64, kernel_size = 5, strides = 2, padding = 'same', activation = 'relu'))

    model[j].add(tf.keras.layers.BatchNormalization())

    model[j].add(tf.keras.layers.Dropout(0.4))



    model[j].add(tf.keras.layers.Conv2D(128, kernel_size = 4, activation = 'relu'))

    model[j].add(tf.keras.layers.BatchNormalization())

    model[j].add(tf.keras.layers.Flatten())

    model[j].add(tf.keras.layers.Dropout(0.4))

    model[j].add(tf.keras.layers.Dense(10, activation = 'softmax'))



    model[j].compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["categorical_accuracy"])
#lower learning rate each epoch

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)



#create list to add training history to

history = [0] * NUM_NETS



for j in range(NUM_NETS):

    #create a validation set to evaluate our model(s) performance

    X_train, X_val, y_train, y_val = train_test_split(train, labels, test_size = 0.1)

    STEPS_PER_EPOCH = X_train.shape[0] // 64

    

    history[j] = model[j].fit_generator(datagen.flow(X_train, y_train, batch_size = BATCH_SIZE),

                                        epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH,  

                                        validation_data = (X_val, y_val), callbacks = [lr_callback],

                                        verbose = VERBOSE)

    

    #display training results

    print(f"CNN {j + 1}: Epochs={EPOCHS}, Train accuracy={max(history[j].history['categorical_accuracy'])}, Validation accuracy={max(history[j].history['val_categorical_accuracy'])}")
#define function to visualize learning curves

def plot_learning_curves(histories): 

    fig, ax = plt.subplots(1, 2, figsize = (20, 10))

    

    #plot accuracies

    for i in range(0, NUM_NETS):

        ax[0].plot(histories[i].history['categorical_accuracy'], color = 'C0')

        ax[0].plot(histories[i].history['val_categorical_accuracy'], color = 'C1')



    #plot losses

    for i in range(0, NUM_NETS):

        ax[1].plot(histories[i].history['loss'], color = 'C0')

        ax[1].plot(histories[i].history['val_loss'], color = 'C1')



    #create legend

    ax[0].legend(['train', 'validation'], loc = 'upper left')

    ax[1].legend(['train', 'validation'], loc = 'upper right')

    

    #set master title

    fig.suptitle("Model Performance", fontsize=14)

    

    #label axis

    ax[0].set_ylabel('Accuracy')

    ax[0].set_xlabel('Epoch')

    ax[1].set_ylabel('Loss')

    ax[1].set_xlabel('Epoch')

    

plot_learning_curves(history)
#average predictions

preds = np.zeros( (test.shape[0],10) ) 

for j in range(NUM_NETS):

    preds += model[j].predict(test) / NUM_NETS

    

#save raw predictions to disk to experiment with further ensembling

probs = pd.DataFrame(preds)

probs.to_csv('ensemble_probs')

probs.columns = probs.columns.astype(str)



print(probs.columns)

probs.head()
#steal ids from sample submission

submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission['Label'] = preds.argmax(axis = 1)

submission.to_csv("ensemble.csv", index = False)

submission.head(10)
#sanity check

submission.shape
prev_cnn_probs = pd.read_csv('../input/mnistsavedprobs/ensemble_probs')

prev_cnn_probs = prev_cnn_probs.drop('Unnamed: 0', axis = 1)      #for formatting reasons

print(prev_cnn_probs.columns)

prev_cnn_probs.head()
#make new probabilites by blending with past training results

new_probs = probs.add(prev_cnn_probs).divide(2)

new_probs.head()
#steal ids from sample submission

submission2 = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission2['Label'] = new_probs.values.argmax(axis = 1)

submission2.to_csv("ensemble2.csv", index = False)

submission2.head(10)
#sanity check

submission2.shape