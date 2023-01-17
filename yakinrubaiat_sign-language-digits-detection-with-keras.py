import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
from keras import optimizers
from os import listdir
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imread, imresize
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline
# Any results you write to the current directory are saved as output.
# Settings:
img_size = 64
grayscale_images = True
num_class = 10
test_size = 0.2
def get_dataset(dataset_path='Dataset'):
    # Getting all data from data path:
    try:
        X = np.load('../input/Sign-language-digits-dataset/X.npy')
        Y = np.load('../input/Sign-language-digits-dataset/Y.npy')
    except:
        labels = listdir(dataset_path) # Geting labels
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X.append(img)
                Y.append(i)
        # Create dateset:
        X = 1-np.array(X).astype('float32')/255.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)
        if not os.path.exists('npy_dataset/'):
            os.makedirs('npy_dataset/')
        np.save('npy_dataset/X.npy', X)
        np.save('npy_dataset/Y.npy', Y)
        
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    plt.imshow(X[702], cmap='gray')
    print(Y[702])
    return X, X_test, Y, Y_test
X_train, X_test , Y_train, Y_test = get_dataset()
classes = 10
plt.imshow(X_train[406], cmap='gray')
print(Y_train[406])
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
X_test = X_test[:,:,:,np.newaxis]
X_train= X_train[:,:,:,np.newaxis]
datagen = ImageDataGenerator(
    rotation_range=16,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12
    )
datagen.fit(X_train)
def KerasModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    model = Sequential()
    model.add(ZeroPadding2D(padding=(3,3), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(1,1), padding = 'valid', strides=(1)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding = 'same', strides=(1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Dropout(0.2))
    
    model.add(ZeroPadding2D(padding=(2,2), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(1,1),padding = 'same', strides=(1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding = 'valid', strides=(1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=4))
    
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(256,activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(10, activation='softmax'))
    ### END CODE HERE ###
    
    return model
model = KerasModel((64,64,1))
model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
model.fit(x = X_train,y = Y_train,epochs =40 ,batch_size = 16 )
model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.Adadelta(),
             metrics=['accuracy'])
model.fit(x = X_train,y= Y_train, batch_size=32,epochs=64)

score = model.evaluate(X_test, Y_test, verbose=0)
preds = model.evaluate(x = X_test,y = Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
test_image = X_test[327]
test_image_array = test_image.reshape(64, 64)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

print(np.round(result,1))
print(Y_test[327])

plt.imshow(test_image_array, cmap='gray')