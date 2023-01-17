import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin')
    return dict

import os
def load_batch_file(batch_filename):
    filepath = os.path.join('../input/cifar-10-batches-py/', batch_filename)
    unpickled = unpickle(filepath)
    return unpickled
train_batch_1 = load_batch_file('data_batch_1')
train_batch_2 = load_batch_file('data_batch_2')
train_batch_3 = load_batch_file('data_batch_3')
train_batch_4 = load_batch_file('data_batch_4')
train_batch_5 = load_batch_file('data_batch_5')
test_batch = load_batch_file('test_batch')
from keras.utils import np_utils
num_classes = 10
train_x = np.concatenate([train_batch_1['data'], train_batch_2['data'], train_batch_3['data'], train_batch_4['data'], train_batch_5['data']])
train_x = train_x.astype('float32') # this is necessary for the division below
train_x /= 255
train_y = np.concatenate([np_utils.to_categorical(labels, num_classes) for labels in [train_batch_1['labels'], train_batch_2['labels'], train_batch_3['labels'], train_batch_4['labels'], train_batch_5['labels']]])
test_x = test_batch['data'].astype('float32') / 255
test_y = np_utils.to_categorical(test_batch['labels'], num_classes)
from keras.models import Sequential
from keras.layers import Dense
img_rows = img_cols = 32
channels = 3
simple_model = Sequential()
simple_model.add(Dense(10_000, input_shape=(img_rows*img_cols*channels,), activation='relu'))
simple_model.add(Dense(10, activation='softmax'))

simple_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
simple_model_history = simple_model.fit(train_x, train_y, batch_size=100, epochs=8, validation_data=(test_x, test_y))
import matplotlib.pyplot as plt
def plot_history(history, title):
    plt.figure(figsize=(10,3))
    # Plot training & validation accuracy values
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
plot_history(simple_model_history, 'Simple NN with 100 batch size')
simple_model_smaller_batch = Sequential()
simple_model_smaller_batch.add(Dense(10_000, input_shape=(img_rows*img_cols*channels,), activation='relu'))
simple_model_smaller_batch.add(Dense(10, activation='softmax'))

simple_model_smaller_batch.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
simple_model_smaller_batch_history = simple_model_smaller_batch.fit(train_x, train_y, batch_size=50, epochs=8, validation_data=(test_x, test_y))
plot_history(simple_model_smaller_batch_history, 'Simple NN with 50 batch size')
simple_model_more_layers = Sequential()
simple_model_more_layers.add(Dense(10_000, input_shape=(img_rows*img_cols*channels,), activation='relu'))
simple_model_more_layers.add(Dense(1_000, activation='relu'))
simple_model_more_layers.add(Dense(100, activation='relu'))
simple_model_more_layers.add(Dense(10, activation='softmax'))

simple_model_more_layers.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
simple_model_more_layers_history = simple_model_more_layers.fit(train_x, train_y, batch_size=100, epochs=8, validation_data=(test_x, test_y))
plot_history(simple_model_more_layers_history, 'Simple NN with more layers')
train_x_reshaped = train_x.reshape(len(train_x), img_rows, img_cols, channels)
test_x_reshaped = test_x.reshape(len(test_x), img_rows, img_cols, channels)
from keras.layers import Conv2D, Flatten
simple_cnn_model = Sequential()
simple_cnn_model.add(Conv2D(32, (3,3), input_shape=(img_rows,img_cols,channels), activation='relu'))
simple_cnn_model.add(Conv2D(32, (3,3), activation='relu'))
simple_cnn_model.add(Conv2D(32, (3,3), activation='relu'))
simple_cnn_model.add(Flatten())
simple_cnn_model.add(Dense(10, activation='softmax'))

simple_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
simple_cnn_model_history = simple_cnn_model.fit(train_x_reshaped, train_y, batch_size=100, epochs=8, validation_data=(test_x_reshaped, test_y))
plot_history(simple_cnn_model_history, 'CNN with 3 convolution layers')
from keras.layers import Dropout
simple_cnn_model_2 = Sequential()
simple_cnn_model_2.add(Conv2D(32, (3,3), input_shape=(img_rows,img_cols,channels), activation='relu'))
simple_cnn_model_2.add(Dropout(0.2))

simple_cnn_model_2.add(Conv2D(32, (3,3), activation='relu'))
simple_cnn_model_2.add(Dropout(0.2))

simple_cnn_model_2.add(Conv2D(32, (3,3), activation='relu'))
simple_cnn_model_2.add(Dropout(0.2))

simple_cnn_model_2.add(Flatten())
simple_cnn_model_2.add(Dense(10, activation='softmax'))

simple_cnn_model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
simple_cnn_model_2_history = simple_cnn_model_2.fit(train_x_reshaped, train_y, batch_size=100, epochs=8, validation_data=(test_x_reshaped, test_y))
plot_history(simple_cnn_model_2_history, '3 convolution layers with dropout')
simple_cnn_model_3 = Sequential()
simple_cnn_model_3.add(Conv2D(64, (3,3), input_shape=(img_rows,img_cols,channels), activation='relu'))
simple_cnn_model_3.add(Dropout(0.2))

simple_cnn_model_3.add(Conv2D(64, (3,3), activation='relu'))
simple_cnn_model_3.add(Dropout(0.2))

simple_cnn_model_3.add(Conv2D(64, (3,3), activation='relu'))
simple_cnn_model_3.add(Dropout(0.2))

simple_cnn_model_3.add(Flatten())
simple_cnn_model_3.add(Dense(128, activation='relu'))
simple_cnn_model_3.add(Dense(10, activation='softmax'))

simple_cnn_model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
simple_cnn_model_3_history = simple_cnn_model_3.fit(train_x_reshaped, train_y, batch_size=100, epochs=8, validation_data=(test_x_reshaped, test_y))
plot_history(simple_cnn_model_3_history, 'CNN with more layers and filters')
