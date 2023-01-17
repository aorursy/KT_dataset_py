from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets.cifar10 import load_batch
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
np.random.seed(0)
import keras

def createModel_1():

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                input_shape=(32,32,3),name="conv1"))
    model.add(keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                name="conv2"))

    model.add(keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool1"))

    model.add(keras.layers.Dropout(rate=0.5, name="d1"))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(128, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                               name="conv3"))
    model.add(keras.layers.Conv2D(128, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                name="conv4"))

    model.add(keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool2"))

    model.add(keras.layers.Dropout(rate=0.5, name="d2"))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(256, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                               name="conv5"))
    model.add(keras.layers.Conv2D(256, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                               name="conv6"))

    model.add(keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool3"))

    model.add(keras.layers.Dropout(rate=0.5, name="d3"))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten(name="flatten"))

    model.add(keras.layers.Dropout(rate=0.5))

    model.add(keras.layers.Dense(128, activation='relu',kernel_initializer='he_normal'))

    model.add(keras.layers.Dropout(rate=0.5))

    model.add(keras.layers.Dense(10, activation='softmax',kernel_initializer='he_normal'))
    
    return model
def createModel_2():
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3,3), kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                input_shape=(32,32,3),name="conv1"))

    model.add(keras.layers.MaxPool2D((2, 2), strides=1, padding='same', name="pool1"))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, (3,3), kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                               name="conv2"))

    model.add(keras.layers.MaxPool2D((2, 2), strides=1, padding='same', name="pool2"))


    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(128, (3,3), kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                               name="conv3"))

    model.add(keras.layers.MaxPool2D((2, 2), strides=1, padding='same', name="pool3"))


    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten(name="flatten"))

    model.add(keras.layers.Dropout(rate=0.5))

    model.add(keras.layers.Dense(10, activation='softmax',kernel_initializer='he_normal'))
    
    return model

def load_data():
    path = r'../input/cifar10-python/cifar-10-batches-py'

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
model = createModel_1()
from keras.optimizers import SGD,Adam,Adagrad
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
(x_train,y_train),(x_test,y_test) = load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            rotation_range=0,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=False)
datagen.fit(x_train)
batch_size = 128
epochs = 200
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
# history = model.fit(x_train,y_train,verbose=1,epochs=epochs,validation_split=0.2,batch_size=batch_size)
history = model.fit_generator(
    datagen.flow(x_train,y_train,batch_size=batch_size),
    steps_per_epoch=x_train.shape[0]//batch_size,
    epochs=epochs,verbose=1,
    validation_data = (x_val,y_val)
)
model.save('cnn_model.h5')
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
test_true = np.argmax(y_test, axis=1)
test_pred = np.argmax(model.predict(x_test), axis=1)
print("CNN Model Accuracy on test set: {:.4f}".format(accuracy_score(test_true, test_pred)))
import matplotlib.pyplot as plt
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
# Plot normalized confusion matrix
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plot_confusion_matrix(test_true, test_pred, classes=classes, normalize=True, title='Normalized confusion matrix')
plt.show()
plt.savefig(r'confusion_matrix.png')
fig, axes = plt.subplots(1,2, figsize=(18, 6))
# Plot training & validation accuracy values

axes[0].plot(history.history['loss'])
axes[0].plot(history.history['val_loss'])
axes[0].set_title('Model loss')
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')

axes[1].plot(history.history['acc'])
axes[1].plot(history.history['val_acc'])
axes[1].set_title('Model accuracy')
axes[1].set_ylabel('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values

plt.show()
plt.savefig('result.png')
result = model.evaluate(x_test,y_test)
print(result)