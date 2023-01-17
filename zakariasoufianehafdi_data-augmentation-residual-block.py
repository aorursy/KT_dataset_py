import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Dropout, Activation, Flatten, Add
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks.callback import EarlyStopping
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
# earlystoping = EarlyStopping(monitor='accuracy', patience=3)

X_input = Input((X_train.shape[1:]))
    
# first conv layer

X = ZeroPadding2D((3,3))(X_input)
X = Conv2D(64, (7,7), name='conv0')(X)
X = BatchNormalization(axis=3, name='bn0')(X)
X = Activation('relu')(X)
X = MaxPooling2D((2,2), name='max_pool0')(X)

X_shortcut = X

# Second conv layer

X = Conv2D(64, (1,1), padding = 'valid', name='conv1')(X)
X = BatchNormalization(axis=3, name='bn1')(X)
X = Activation('relu')(X)

# Third layer

X = Conv2D(256, (3,3), padding = 'same', name='conv2')(X)
X = BatchNormalization(axis=3, name='bn2')(X)
X = Activation('relu')(X)

# Fourth layer 

X = Conv2D(64, (1,1), padding = 'valid', name='conv3')(X)
X = BatchNormalization(axis=3, name='bn3')(X)

# Residual block to keep the identity
X = Add()([X, X_shortcut])

X = Activation('relu')(X)

X = MaxPooling2D((2,2), name='max_pool1')(X)

# Flatten X

X= Flatten()(X)

# Fully connected layer
# X = Dropout(0.2)(X)

X = Dense(1, activation='sigmoid', name='fc')(X)

# Creating Model

model = Model(X_input, X, name='happy_model')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fitting with initial data
model.fit(X_train, Y_train, batch_size=16, epochs=20)
# Predicting with initial data
preds = model.evaluate(X_test, Y_test)

print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
# Generate more data
generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                              horizontal_flip=True, zoom_range=0.1)
# Concatenate intial data with augmented data
X_train_aug = X_train
Y_train_aug = Y_train
tupl = generator.flow(X_train, Y_train)

for i in range(len(tupl)):
    X_train_aug = np.concatenate((X_train_aug, tupl[i][0]), axis=0)
    Y_train_aug = np.concatenate((Y_train_aug, tupl[i][1]), axis=0)

print ("number of training examples = " + str(X_train_aug.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train_aug.shape))
print ("Y_train shape: " + str(Y_train_aug.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
# fitting with augmented data
model.fit(X_train_aug, Y_train_aug, batch_size=16, epochs=20)
# predicting with augmented data
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
