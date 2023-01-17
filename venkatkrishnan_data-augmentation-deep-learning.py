import numpy as np

import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load data method loads the dataset into 2 tuples
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
# Count of the X_train and X_test
print("X train {} and Test size {}".format(X_train.shape[0], X_test.shape[0]))
plt.figure(figsize=(10,10))

for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i])
    plt.axis('off')
plt.show()

print('Label %s' % (y_test[0:5]))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

X_train.shape
# Convert to the tensor shape
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

print("X_Train shape {}".format(X_train.shape))
print("X_Test shape {}".format(X_test.shape))
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
def build_model():
    """
        Method constructs the CNN architecture with 2D-Convolution
    """
    model = keras.models.Sequential()
    
    model.add(keras.layers.Convolution2D(32, 3, 3, input_shape=(28, 28, 1)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.25))
    
    model.add(keras.layers.Convolution2D(32, 3, 3))
    model.add(keras.layers.Activation('relu'))
    
    model.add(keras.layers.Convolution2D(32, 3, 3))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.25))
    
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Activation('relu'))
    
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Activation('softmax'))
    
    return model
    
    
cnn_model = build_model()

cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn_model.summary()
history = cnn_model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
cnn_model.evaluate(X_test, y_test)
history_dict = history.history

print(history_dict.keys())
def plot_train_instrumentation(epochs, data, train_param, val_param):
    
    plt.figure(figsize=(10,7))
    
    plt.plot(epochs, data[train_param], 'g', label=f'Training ({train_param})')
    plt.plot(epochs, data[val_param], 'red', label=f'Validation ({val_param})')
    
    plt.title("Training performance")
    plt.xlabel('Epochs')
    plt.ylabel(train_param)
    
    plt.legend()
    plt.show()
epochs = range(1, len(history_dict['accuracy'])+1)

plot_train_instrumentation(epochs, history_dict, 'accuracy', 'val_accuracy')
plot_train_instrumentation(epochs, history_dict, 'loss', 'val_loss')
# Create data augmentation object
data_augmentor = ImageDataGenerator(rotation_range=50, 
                                    width_shift_range=0.01, 
                                    height_shift_range=0.01)

# fit the training data
data_augmentor.fit(X_train)

augment = data_augmentor.flow(X_train[1:2], batch_size=1)

for i in range(1, 6):
    plt.subplot(1,5,i)
    plt.imshow(augment.next().squeeze())
    plt.axis('off')
plt.show()
history_data_aug = cnn_model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
cnn_model.evaluate(X_test, y_test)
history_dict = history_data_aug.history
plot_train_instrumentation(epochs, history_dict, 'accuracy', 'val_accuracy')
plot_train_instrumentation(epochs, history_dict, 'loss', 'val_loss')
def construct_model(input_shape=(28,28,1)):
    
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape= input_shape))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32, kernel_size=5, strides=2, padding='same',  activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))
    ## Dropout Regularization of 0.4 in order to avoid overfitting
    model.add(keras.layers.Conv2D(64, kernel_size=3,  activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, kernel_size=3,  activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same',  activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))
    
    model.add(keras.layers.Conv2D(64, kernel_size=4,  activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model
    
conv_model = construct_model((28,28,1))

conv_model.summary()
# Compile the model
conv_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model with training set
history = conv_model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
def plot_model_performance(model_history, metric, val_metric):
    plt.figure(figsize=(10,8))
    plt.plot(model_history.history[metric], label=str('Training '+ metric))
    plt.plot(model_history.history[val_metric], label=str('Validation '+ val_metric))
    plt.title(metric+" vs "+val_metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.show()
plot_model_performance(history, 'accuracy', 'val_accuracy')
plot_model_performance(history, 'loss', 'val_loss')
