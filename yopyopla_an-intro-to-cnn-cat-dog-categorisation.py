import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import rmsprop
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications import VGG16
from PIL import Image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import os
print(os.listdir("../input/cifar10catdog/data/data"))
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

epochs = 10
batch_size = 32

#input_dim = (100, 100, 3)

def build_CNN(input_dim, output_dim, lr=0.001):
    classifier  = Sequential()

    # Add 2 convolution layers
    #input dim : 32x32x3
    classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=input_dim, activation='relu'))
    # Add pooling layer
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    # Add 2 more convolution layers
    classifier.add(Conv2D(filters=60, kernel_size=(3,3), activation='relu'))
    # Add pooling layer
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    # Add 2 more convolution layers
    classifier.add(Conv2D(filters=120, kernel_size=(3,3), activation='relu'))
    # Add pooling layer
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    classifier.add(Flatten())
    classifier.add(Dense(units=150, activation='relu'))
    
    classifier.add(Dense(units=output_dim, activation='softmax'))
    
    # Compiling the ANN
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return classifier


data_gen = ImageDataGenerator(        
        rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = data_gen.flow_from_directory(
        '../input/cifar10catdog/data/data/train',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '../input/cifar10catdog/data/data/valid',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical')


%%time


BACTH_SIZE = 32
classifier = build_CNN((32,32,3), 2, lr=0.001)
history = classifier.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // BACTH_SIZE,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames) // BACTH_SIZE)
def plotLog(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
plotLog(history)
epochs = 10
batch_size = 32
d_cnn =  0.25
d_fc = 0.25

def build_CNN(input_dim, output_dim, lr):
    classifier  = Sequential()

    # Add 2 convolution layers
    #input dim : 32x32x3
    classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=input_dim, activation='relu'))
    # Add pooling layer
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    classifier.add(Dropout(d_cnn))
    
    # Add 2 more convolution layers
    classifier.add(Conv2D(filters=60, kernel_size=(3,3), activation='relu'))
    # Add pooling layer
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    classifier.add(Dropout(d_cnn))
    
    # Add 2 more convolution layers
    classifier.add(Conv2D(filters=120, kernel_size=(3,3), activation='relu'))
    # Add pooling layer
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    classifier.add(Dropout(d_fc))
    
    classifier.add(Flatten())
    classifier.add(Dense(units=150, activation='relu'))
    
    classifier.add(Dense(units=output_dim, activation='softmax'))
    
    # Compiling the ANN
    opti = rmsprop(lr=lr)
    
    classifier.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return classifier
%%time
BACTH_SIZE = 32
classifier = build_CNN((32,32,3), 2, lr=0.001)
history1 = classifier.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // BACTH_SIZE,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames) // BACTH_SIZE)
plotLog(history1)


data_gen = ImageDataGenerator(        
        horizontal_flip=True,
        
        rotation_range=5,
        width_shift_range=0.02,
        height_shift_range=0.10,
    
        fill_mode='nearest',
        rescale=1./255)

test_datagen_2 = ImageDataGenerator(rescale=1./255)

train_generator_2 = data_gen.flow_from_directory(
        '../input/cifar10catdog/data/data/train',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical')

validation_generator_2 = test_datagen.flow_from_directory(
        '../input/cifar10catdog/data/data/valid',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical')


%%time
BACTH_SIZE = 32
classifier = build_CNN((32,32,3), 2, lr=0.001)
history2 = classifier.fit_generator(
    train_generator_2,
    steps_per_epoch=len(train_generator_2.filenames) // BACTH_SIZE,
    epochs=10,
    validation_data=validation_generator_2,
    validation_steps=len(validation_generator_2.filenames) // BACTH_SIZE)
plotLog(history2)
%%time
weights_path = "../input/pretrained-models/pretrained_models/model_categorical_complex.h5"
BACTH_SIZE = 32

classifier = build_CNN((32,32,3), 2, lr=0.001)

classifier.load_weights(weights_path)

 
# Check the trainable status of the individual layers
for layer in classifier.layers:
    print(layer, layer.trainable)

    
history3 = classifier.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // BACTH_SIZE,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames) // BACTH_SIZE)


plotLog(history3)
