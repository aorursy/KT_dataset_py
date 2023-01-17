# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
from glob import glob

imagePatches = glob('../input/chest-xray-pneumonia/chest_xray/chest_xray/**/**/*.jpeg', recursive=False)

print(len(imagePatches))
import cv2

import fnmatch

pattern_normal = '*NORMAL*'

pattern_bacteria = '*_bacteria_*'

pattern_virus = '*_virus_*'



normal = fnmatch.filter(imagePatches, pattern_normal)

bacteria = fnmatch.filter(imagePatches, pattern_bacteria)

virus = fnmatch.filter(imagePatches, pattern_virus)

x = []

y = []

for img in imagePatches:

    full_size_image = cv2.imread(img)

    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)

    im = im.astype(np.float32)/255.

    x.append(im)

    if img in normal:

        y.append(0)

    elif img in bacteria:

        y.append(1)

    elif img in virus:

        y.append(1)

    else:

        #break

        print('no class')

x = np.array(x)

y = np.array(y)
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 101, stratify=y)

y_train = to_categorical(y_train, num_classes = 2)

y_valid = to_categorical(y_valid, num_classes = 2)

del x,y
import matplotlib.pyplot as plt

fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))

for (c_x, c_y, c_ax) in zip(x_valid, y_valid, m_axs.flatten()):

    c_ax.imshow(c_x[:,:,0], cmap = 'bone')

    c_ax.set_title(str(c_y))

    c_ax.axis('off')
# config the session 

import tensorflow as tf



# Set the seed for hash based operations in python

os.environ['PYTHONHASHSEED'] = '0'



# Set the numpy seed

np.random.seed(111)



# Set the random seed in tensorflow at graph level

tf.random.set_seed(111)

from random import seed

seed(111)
from keras import layers, Model, backend

channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

def model():

    img_input = layers.Input(shape = (224, 224, 3))

    x = layers.Conv2D(32, (3,3),

                      padding = 'same', use_bias = False,

                      name = 'block1_conv1')(img_input)

    x = layers.BatchNormalization(axis = channel_axis, name = 'block1_bn1')(x)

    x = layers.Activation('relu', name = 'block1_act1')(x)

    x = layers.Conv2D(32, (3,3),

                      padding = 'same', use_bias = False,

                      name = 'block1_conv2')(x)

    x = layers.BatchNormalization(axis = channel_axis, name = 'block1_bn2')(x)

    x = layers.Activation('relu', name = 'block1_act2')(x)

    x = layers.MaxPooling2D((2, 2),

                            strides=(2, 2),

                            padding='same',

                            name='block1_pool')(x)



    # block 2

    x = layers.Conv2D(64, (3,3),

                      padding = 'same', use_bias = False,

                      name = 'block2_conv1')(x)

    x = layers.BatchNormalization(axis = channel_axis, name = 'block2_bn1')(x)

    x = layers.Activation('relu', name = 'block2_act1')(x)

    x = layers.Conv2D(64, (3,3),

                      padding = 'same', use_bias = False,

                      name = 'block2_conv2')(x)

    x = layers.BatchNormalization(axis = channel_axis, name = 'block2_bn2')(x)

    x = layers.Activation('relu', name = 'block2_act2')(x)

    x = layers.MaxPooling2D((2, 2),

                            strides=(2, 2),

                            padding='same',

                            name='block2_pool')(x)



    # block 3

    x = layers.Conv2D(128, (3,3),

                      padding = 'same', use_bias = False,

                      name = 'block3_conv1')(x)

    x = layers.BatchNormalization(axis = channel_axis, name = 'block3_bn1')(x)

    x = layers.Activation('relu', name = 'block3_act1')(x)

    x = layers.Conv2D(128, (3,3),

                      padding = 'same', use_bias = False,

                      name = 'block3_conv2')(x)

    x = layers.BatchNormalization(axis = channel_axis, name = 'block3_bn2')(x)

    x = layers.Activation('relu', name = 'block3_act2')(x)

    x = layers.MaxPooling2D((3, 3),

                            strides=(3, 3),

                            padding='same',

                            name='block3_pool')(x)



  # block 4

    x = layers.Conv2D(512, (3,3),

                      padding = 'same', use_bias = False,

                      name = 'block4_conv1')(x)

    x = layers.BatchNormalization(axis = channel_axis, name = 'block4_bn1')(x)

    x = layers.Activation('relu', name = 'block4_act1')(x)

    x = layers.Conv2D(512, (3,3),

                      padding = 'same', use_bias = False,

                      name = 'block4_conv2')(x)

    x = layers.BatchNormalization(axis = channel_axis, name = 'block4_bn2')(x)

    x = layers.Activation('relu', name = 'block4_act2')(x)

    x = layers.MaxPooling2D((3, 3),

                            strides=(3, 3),

                            padding='same',

                            name='block4_pool')(x)

    x = layers.Flatten(name='flatten')(x)

    x = layers.Dense(512, activation='relu', name='fc1')(x)

    x = layers.Dense(64, activation='relu', name='fc2')(x)

    x = layers.Dense(2, activation='softmax', name='predictions')(x)

    model = Model(inputs=img_input, outputs=x, name = 'own_build_model')

    return model

model = model()

model.summary()
LEARN_RATE = 1e-4

from keras.optimizers import Adam

model.compile(optimizer = Adam(lr = LEARN_RATE), loss = 'categorical_crossentropy',

                           metrics = ['categorical_accuracy'])
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}.best_only.hdf5".format('save')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)



# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, 

#                                    patience=10, verbose=1, mode='auto', 

#                                    epsilon=0.0001, cooldown=5, min_lr=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=15) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early]
history = model.fit(x_train,y_train,batch_size = 32, 

                    epochs = 50, verbose=1,  validation_split=0.2, callbacks=callbacks_list)
# summarize history for accuracy

plt.plot(history.history['categorical_accuracy'])

plt.plot(history.history['val_categorical_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.load_weights(weight_path)

model.save('full_model.h5')
test_loss, test_score = model.evaluate(x_valid, y_valid, batch_size=24)

print("Loss on test set: ", test_loss)

print("Accuracy on test set: ", test_score)
pred_y = model.predict(x_valid, callbacks=callbacks_list)
# Original labels

orig_test_labels = np.argmax(y_valid, axis=-1)



print(orig_test_labels.shape)

print(pred_y.shape)
from sklearn.metrics import classification_report

print(classification_report(np.argmax(y_valid, axis = 1),np.argmax(pred_y, axis = 1)))
from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

# Get the confusion matrix

cm  = confusion_matrix(np.argmax(y_valid, axis = 1), np.argmax(pred_y, axis = 1))

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
# Calculate Precision and Recall

tn, fp, fn, tp = cm.ravel()



precision = tp/(tp+fp)

recall = tp/(tp+fn)



print("Recall of the model is {:.2f}".format(recall))

print("Precision of the model is {:.2f}".format(precision))
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(np.argmax(y_valid,-1)==0, pred_y[:,0])

fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 250)

ax1.plot(fpr, tpr, 'b.-', label = 'Own-Model (AUC:%2.2f)' % roc_auc_score(np.argmax(y_valid,-1)==0, pred_y[:,0]))

ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')

ax1.legend(loc = 4)

ax1.set_xlabel('False Positive Rate')

ax1.set_ylabel('True Positive Rate');

ax1.set_title('Pneumonia Classification ROC Curve')

fig.savefig('roc_valid.pdf')