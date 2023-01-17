import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Dropout, Activation, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import layer_utils
# from keras.utils.data_utils import get_file
# from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K
K.set_image_data_format('channels_last')
import h5py
train_dataset = h5py.File('/kaggle/input/happy-datasets/train_happy.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

test_dataset = h5py.File('/kaggle/input/happy-datasets/test_happy.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

classes = np.array(test_dataset["list_classes"][:]) # the list of classes

train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

# Normalize image vectors
X_train = train_set_x_orig/255.
X_test = test_set_x_orig/255.
# Reshape
Y_train = train_set_y_orig.T
Y_test = test_set_y_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
def HappyModel(input_shape):
    # Input Layer
    X_input = Input((input_shape))
    
    # first conv layer
    
    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(32, (7,7), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name='max_pool0')(X)
    
    # Dropout layer
    
    X = Dropout(0.2)(X)
    
    # Second conv layer
    
    X = ZeroPadding2D((1,1))(X)
    X = Conv2D(32, (3,3), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name='max_pool1')(X)
    
    # Flatten X
    
    X= Flatten()(X)
    
    # Fully connected layer
    
    X = Dense(1, activation='sigmoid', name='fc')(X)
    
    # Calling Model
    
    model = Model(X_input, X, name='happy_model')
    
    return model
# Creating the Model :

happy_model = HappyModel(X_train.shape[1:])
# Compiling the Model:

happy_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Data Augmentation

Data_aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                              horizontal_flip=True, zoom_range=0.1)

Data_aug.fit(X_train)
# Fitting the Model

happy_model.fit_generator(Data_aug.flow(X_train, Y_train, batch_size=20),
                    steps_per_epoch=len(X_train) / 32, epochs=40)
# Testing

preds = happy_model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
# An exemple of augmentation data

generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                              horizontal_flip=True, zoom_range=0.1)

x = generator.flow(X_train[0:1,:])


plt.figure()
plt.imshow(X_train[0,:])
plt.title('original')
plt.axis('off')
plt.figure()
plt.imshow(x[0][0])
plt.title('augmented')
plt.axis('off')
plt.show()

