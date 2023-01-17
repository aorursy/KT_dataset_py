# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import tensorflow as tf

import keras

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
display(train.shape,test.shape)
display(train.columns, test.columns)
X = train.drop(['label'],axis=1)

y = train['label']

display(X.head(),y.head())
# from keras.utils import to_categorical

# num_classes = len(set(y))

# y = to_categorical(y, num_classes=num_classes)

# y
test_id = test.loc[:,'id']

test = test.iloc[:, 1:]
X = np.array(X).reshape(len(X),28,28)

y = np.array(y)

test = np.array(test).reshape(len(test),28,28)

display(X.shape,test.shape)
plt.imshow(X[24])
import matplotlib.pyplot as plt

fig = plt.figure()

for i in range(9):

  plt.subplot(3,3,i+1)

  plt.tight_layout()

  plt.imshow(X[i], cmap='gray', interpolation='none')

  plt.title("Digit: {}".format(y[i]))

  plt.xticks([])

  plt.yticks([])
# from keras.applications.vgg16 import VGG16

# from keras.applications.resnet import ResNet50

# model = ResNet50(include_top=False, input_shape=(300, 300, 3))
#Image augumantation

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

# def train_mnist():





# #     class myCallback(tf.keras.callbacks.Callback):

# #         def on_epoch_end(self,epoch,logs={}):

# #             if(logs.get('acc')>0.999) and (logs.get('val_acc')>0.999):

# #                 print("\nReached 99.9% accuracy so cancelling training!")

# #                 self.model.stop_training = True    

# #     callbacks = myCallback()



#     callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



#     X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0,test_size=0.2,shuffle=True)

#     X_val, X_test, y_val, y_test = train_test_split(X_val,y_val,random_state=0,test_size=0.5,shuffle=True)



#     X_train, X_val = (X_train / 255.0).reshape(len(X_train),28,28,1), (X_val / 255.0).reshape(len(X_val),28,28,1)





#     datagen.fit(X_train)

    

#     model = tf.keras.models.Sequential([



#         tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),

# #         tf.keras.layers.BatchNormalization(),

#         tf.keras.layers.MaxPooling2D(2, 2),

#         tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

# #         tf.keras.layers.BatchNormalization(),

#         tf.keras.layers.MaxPooling2D(2,2),

        

#         tf.keras.layers.Flatten(),

#         tf.keras.layers.Dense(1024, activation='relu'),

#         tf.keras.layers.Dropout(rate=0.5),

#         tf.keras.layers.Dense(1024, activation='relu'),

#         tf.keras.layers.Dropout(rate=0.5),

#         tf.keras.layers.Dense(10, activation='softmax')



#     ])



#     #optimizer = tf.keras.optimizers.Adam(lr=0.001)

#     optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9,decay=1e-6)

#     model.compile(optimizer=optimizer,

#                   loss='sparse_categorical_crossentropy',

#                   metrics=['accuracy'])

    

#     batch_size = 32

#     history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

#                               validation_data = (X_val,y_val), 

#                               epochs=100,

#                               steps_per_epoch=X_train.shape[0] // batch_size,

#                               callbacks=[callback])



#     return history, model, X_test, y_test
def train_mnist():





#     class myCallback(tf.keras.callbacks.Callback):

#         def on_epoch_end(self,epoch,logs={}):

#             if(logs.get('acc')>0.999) and (logs.get('val_acc')>0.999):

#                 print("\nReached 99.9% accuracy so cancelling training!")

#                 self.model.stop_training = True    

#     callbacks = myCallback()



    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



    X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0,test_size=0.2,shuffle=True)

    X_val, X_test, y_val, y_test = train_test_split(X_val,y_val,random_state=0,test_size=0.5,shuffle=True)



    X_train, X_val = (X_train / 255.0).reshape(len(X_train),28,28,1), (X_val / 255.0).reshape(len(X_val),28,28,1)





    datagen.fit(X_train)

    

    model = tf.keras.models.Sequential([



        tf.keras.layers.Conv2D(64, (3,3), input_shape=(28, 28, 1)),

#         tf.keras.layers.BatchNormalization(),

        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3,3)),

#         tf.keras.layers.BatchNormalization(),

        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.MaxPooling2D(2,2),

        

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1024),

        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Dense(1024),

        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Dense(10, activation='softmax')



    ])



    #optimizer = tf.keras.optimizers.Adam(lr=0.001)

    optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9,decay=1e-6)

    model.compile(optimizer=optimizer,

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])

    

    batch_size = 32

    history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              validation_data = (X_val,y_val), 

                              epochs=60,

                              steps_per_epoch=X_train.shape[0] // batch_size,

                              callbacks=[callback])



    return history, model, X_test, y_test
# def train_mnist():

    

#     callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



#     X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0,test_size=0.2,shuffle=True)

#     X_val, X_test, y_val, y_test = train_test_split(X_val,y_val,random_state=0,test_size=0.5,shuffle=True)

#     X_train, X_val = (X_train / 255.0).reshape(len(X_train),28,28,1), (X_val / 255.0).reshape(len(X_val),28,28,1)



#     datagen.fit(X_train)

    

#     model = tf.keras.models.Sequential([

        

#         tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)),

#         tf.keras.layers.BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"),

#         tf.keras.layers.LeakyReLU(alpha=0.1),

#         tf.keras.layers.Conv2D(64,  (3,3), padding='same'),

#         tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),

#         tf.keras.layers.LeakyReLU(alpha=0.1),



#         tf.keras.layers.MaxPooling2D(2, 2),

#         tf.keras.layers.Dropout(0.2),

    

#         tf.keras.layers.Conv2D(128, (3,3), padding='same'),

#         tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),

#         tf.keras.layers.LeakyReLU(alpha=0.1),

#         tf.keras.layers.Conv2D(128, (3,3), padding='same'),

#         tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),

#         tf.keras.layers.LeakyReLU(alpha=0.1),

    

#         tf.keras.layers.MaxPooling2D(2,2),

#         tf.keras.layers.Dropout(0.2),    

    

#         tf.keras.layers.Conv2D(256, (3,3), padding='same'),

#         tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),

#         tf.keras.layers.LeakyReLU(alpha=0.1),

#         tf.keras.layers.Conv2D(256, (3,3), padding='same'),

#         tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),

#         tf.keras.layers.LeakyReLU(alpha=0.1),



    

#         tf.keras.layers.MaxPooling2D(2,2),

#         tf.keras.layers.Dropout(0.2),

    

    

#         tf.keras.layers.Flatten(),

#         tf.keras.layers.Dense(256),

#         tf.keras.layers.LeakyReLU(alpha=0.1),



#         tf.keras.layers.BatchNormalization(),

#         tf.keras.layers.Dense(10, activation='softmax')

    

        

#     ])

    

    

#     optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9,decay=1e-6)

#     model.compile(optimizer=optimizer,

#                   loss='sparse_categorical_crossentropy',

#                   metrics=['accuracy'])

    

#     batch_size = 32

#     history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

#                               validation_data = (X_val,y_val), 

#                               epochs=1,

#                               steps_per_epoch=X_train.shape[0] // batch_size,

#                               callbacks=[callback])



#     return history, model, X_test, y_test
history, model, X_test, y_test = train_mnist()
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
# pd.DataFrame(history.history).plot(figsize=(8, 5))

# plt.grid(True)

# plt.gca().set_ylim(0.75, 1.) # set the vertical range

# plt.show()



# pd.DataFrame(history.history).plot(figsize=(8, 5))

# plt.grid(True)

# plt.gca().set_ylim(0.0, 0.4) # set the vertical range

# plt.show()
X_test = (X_test / 255.0).reshape(len(X_test),28,28,1)

test = (test / 255.0).reshape(len(test),28,28,1)
loss, acc = model.evaluate(X_test,y_test)

print(f"Accuracy: {acc}")

print(f"Loss: {loss}")
from sklearn.metrics import confusion_matrix

y_pred = model.predict_classes(X_test)

CM = confusion_matrix(y_test, y_pred)



from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))

plt.show()
predictions = model.predict_classes(test)

display(predictions,len(predictions))
df = pd.DataFrame({'id': test_id,

                   'Label': predictions})
df
df.to_csv('submission.csv', index=False)