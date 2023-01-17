# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense , Conv2D,MaxPool2D,Flatten ,Dropout,BatchNormalization

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop,Adam,Adagrad,Adamax,Adadelta

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping



from sklearn.metrics import classification_report ,confusion_matrix



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data_raw=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data_raw=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_data=test_data_raw.copy(deep=True)

train_data=train_data_raw.copy(deep=True)
train_data.head()
train_data.shape
train_data.isna().any().sum()
train_X=train_data.drop(columns=['label'])

train_y=train_data['label']
train_X = np.array(train_X)

train_y = np.array(train_y)

test=np.array(test_data)
train_X = train_X / 255.0

test= test / 255.0
train_X=train_X.reshape(-1,28,28,1)

test=test.reshape(-1,28,28,1)
train_y = to_categorical(train_y, num_classes = len(np.unique(train_data["label"])))
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=42, stratify= train_y, shuffle=True)
epochs = 50

batch_size = 16
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(10, activation = "softmax"))
model.summary()
optimizer = Adamax()

model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
lr_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

es = EarlyStopping(monitor='val_loss',

                              min_delta=0,

                              patience=5,

                              verbose=0, mode='auto')
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.12,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.12,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
# Fit the model

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,y_test), steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[lr_reduction, es], shuffle=True)
model.evaluate(X_train, y_train), model.evaluate(X_test, y_test)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
ypred = model.predict(X_test)

ypred = np.argmax(ypred, axis=1)

ytest = np.argmax(y_test, axis=1)



cf_matrix = confusion_matrix(ytest, ypred)



plt.figure(figsize=(20,8))

ax = sns.heatmap(cf_matrix, annot=True, fmt='g')

plt.show()



print("\n\n")

print(classification_report(ytest, ypred))
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("digit_reconizer_submission.csv",index=False)