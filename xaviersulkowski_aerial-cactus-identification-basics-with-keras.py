import os

import shutil

import zipfile

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt





from cv2 import imread

from IPython.display import Image



from keras import optimizers

from keras import regularizers

from keras import layers,models

from keras.preprocessing import image

from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.imagenet_utils import preprocess_input



from keras import backend as K
def unzip(path):

    with zipfile.ZipFile(path,"r") as z:

        z.extractall('.')



# since Keras remove f1-score from metrics it's need to be calculated manually        



def my_recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def my_precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def my_f1(y_true, y_pred):

    precision = my_precision(y_true, y_pred)

    recall = my_recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
y_train=pd.read_csv('../input/aerial-cactus-identification/train.csv')

y_test=pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')



train_zip_path = '../input/aerial-cactus-identification/train.zip'

test_zip_path = '../input/aerial-cactus-identification/test.zip'



train_path = './train'

test_path = './test'



unzip(train_zip_path)

unzip(test_zip_path)
print(y_train.head(5))
ax = sns.countplot(x="has_cactus", data=y_train, palette=sns.color_palette("coolwarm", 2))
fig,ax = plt.subplots(1,5,figsize=(15,3))



for i, idx in enumerate(y_train[y_train['has_cactus']==1]['id'][:5]):

  path = os.path.join(train_path,idx)

  ax[i].imshow(imread(path))

    

fig.suptitle('pictures with cactus')
fig,ax = plt.subplots(1,5,figsize=(15,3))



for i, idx in enumerate(y_train[y_train['has_cactus']==0]['id'][:5]):

  path = os.path.join(train_path,idx)

  ax[i].imshow(imread(path))

    

fig.suptitle('pictures without cactus')
datagen=ImageDataGenerator(rescale=1./255)  # if we rescale like this (1/255) we turn images to garyscale 

batch_size=150
validation_size = 0.3

split_idx = int(len(y_train) * (1 - validation_size))



y_train.has_cactus = y_train.has_cactus.astype(str)



train_generator = datagen.flow_from_dataframe(dataframe=y_train[:split_idx],

                                              directory=train_path,

                                              x_col='id',

                                              y_col='has_cactus',

                                              class_mode='binary',

                                              batch_size=batch_size,

                                              target_size=(150,150)

                                             )





validation_generator = datagen.flow_from_dataframe(dataframe=y_train[split_idx:],

                                                   directory=train_path,

                                                   x_col='id',

                                                   y_col='has_cactus',

                                                   class_mode='binary',

                                                   batch_size=50,

                                                   target_size=(150,150)

                                                  )
model=models.Sequential()

model.add(layers.ZeroPadding2D((1,1),input_shape=(150,150,3)))

model.add(layers.Conv2D(32,(3,3),activation='relu'))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.ZeroPadding2D((1,1)))



model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.ZeroPadding2D((1,1)))



model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.ZeroPadding2D((1,1)))



model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.ZeroPadding2D((1,1)))



model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dropout(0.7))

model.add(layers.Dense(1,activation='sigmoid'))   # for more than two calsses use softmax instead of sigmoid

         



model.summary()
model.compile(loss='binary_crossentropy',

              optimizer=optimizers.Adam(),

              metrics=['acc', my_f1]

             )
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
epochs = 15

history = model.fit_generator(train_generator,

                              steps_per_epoch=100,

                              epochs=epochs,

                              validation_data=validation_generator,

                              validation_steps=50,

                              callbacks=[es]

                             )
acc = history.history['acc']  ##getting  accuracy of each epochs

epochs_ = range(0,len(acc))    

plt.plot(epochs_,acc,label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val = history.history['val_acc']  ##getting validation accuracy of each epochs

plt.scatter(epochs_,acc_val,label="validation accuracy", c="r", alpha=0.5)

plt.title("no of epochs vs accuracy")

plt.legend()



f1 = history.history['my_f1']  ##getting  accuracy of each epochs

epochs_ = range(0,len(f1))    

plt.plot(epochs_,f1,label='training f1-score')

plt.xlabel('no of epochs')

plt.ylabel('f1-score')



f1_val = history.history['val_my_f1']  ##getting validation accuracy of each epochs

plt.scatter(epochs_,f1_val,label="validation f1-score", c="r", alpha=0.5)

plt.title("no of epochs vs f1-score")

plt.legend()

loss = history.history['loss']    ##getting  loss of each epochs

epochs_ = range(0,len(loss))

plt.plot(epochs_,loss,label='training loss')

plt.xlabel('No of epochs')

plt.ylabel('loss')



loss_val = history.history['val_loss']  ## getting validation loss of each epochs

plt.scatter(epochs_,loss_val,label="validation loss",  c="r", alpha=0.5)

plt.title('no of epochs vs loss')

plt.legend()
y_test.has_cactus = y_test.has_cactus.astype(str)



test_generator = datagen.flow_from_dataframe(dataframe=y_test,

                                             directory=test_path,

                                             x_col='id',

                                             y_col='has_cactus',

                                             class_mode=None,  # only data, no labels

                                             shuffle=False,

                                             target_size=(150,150)

                                             )



y_pred = model.predict_generator(test_generator, verbose=1)



shutil.rmtree(train_path)

shutil.rmtree(test_path)



y_test['has_cactus'] = y_pred

y_test.to_csv('submission.csv', index = False)