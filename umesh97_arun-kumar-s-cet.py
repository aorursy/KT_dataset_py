import datetime as dt
import pandas as pd
import numpy as np
start = dt.datetime.now()
# Load npz file containing image arrays
x_npz = np.load("../input/resize-and-save-images-as-numpy-arrays-128x128/x_images_arrays.npz")
x = x_npz['arr_0']

# Load binary encoded labels for Lung Infiltrations: 0=Not_infiltration 1=Infiltration
y_npz = np.load("../input/resize-and-save-images-as-numpy-arrays-128x128/y_infiltration_labels.npz")
y = y_npz['arr_0']
from sklearn.model_selection import train_test_split

# First split the data in two sets, 80% for training, 20% for Val/Test)
X_train, X_valtest, y_train, y_valtest = train_test_split(x,y, test_size=0.2, random_state=1, stratify=y)

# Second split the 20% into validation and test sets
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1, stratify=y_valtest)
print(np.array(X_train).shape)
print(np.array(X_val).shape)
print(np.array(X_test).shape)
# Import the VGG16 network architecture
from keras.applications import VGG16;

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_DEPTH = 3
BATCH_SIZE = 16

# Instantiate the model with the pre-trained weights (no top)
conv_base = VGG16(weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                  include_top=False, 
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

# Show the architecture
conv_base.summary()
# Extract features
train_features = conv_base.predict(np.array(X_train), batch_size=BATCH_SIZE, verbose=1)
test_features = conv_base.predict(np.array(X_test), batch_size=BATCH_SIZE, verbose=1)
val_features = conv_base.predict(np.array(X_val), batch_size=BATCH_SIZE, verbose=1)
# Save extracted features
np.savez("train_features", train_features, y_train)
np.savez("test_features", test_features, y_test)
np.savez("val_features", val_features, y_val)
# Current shape of features
print(train_features.shape, "\n",  test_features.shape, "\n", val_features.shape)
# Flatten extracted features
train_features_flat = np.reshape(train_features, (4484, 4*4*512))
test_features_flat = np.reshape(test_features, (561, 4*4*512))
val_features_flat = np.reshape(val_features, (561, 4*4*512))
from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
# Define the densely connected classifier
NB_TRAIN_SAMPLES = train_features_flat.shape[0]
NB_VALIDATION_SAMPLES = val_features_flat.shape[0]
NB_EPOCHS = 50

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=(4*4*512)))
#model.add(layers.Dropout(0.1))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

reduce_learning = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='auto',
    epsilon=0.0001,
    cooldown=2,
    min_lr=0)

eary_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=7,
    verbose=1,
    mode='auto')

callbacks = [reduce_learning, eary_stopping]
# Train the the model
history = model.fit(
    train_features_flat,
    y_train,
    epochs=NB_EPOCHS,
    validation_data=(val_features_flat, y_val),
    callbacks=callbacks
)
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()
# Run time
end = dt.datetime.now()
print("Run time:", (end - start).seconds, 'seconds')