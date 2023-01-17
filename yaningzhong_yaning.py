import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential, save_model, load_model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint

sns.set(style='white', context='notebook', palette='deep')
### load data

train = pd.read_csv('../input/training/training.csv')

test = pd.read_csv('../input/test/test.csv')

sample = pd.read_csv('../input/SampleSubmission.csv')

look_id = pd.read_csv('../input/IdLookupTable.csv')
train.describe()
train.tail().T
train.fillna(method='ffill', inplace=True)

train.tail().T
train.isnull().any().describe()
Img = []

for i in range(7049):

    img = train["Image"][i].split(' ')

    img = ['0' if x=='' else x for x in img]

    Img.append(img)
Img = np.array(Img, dtype='float')

X_train = Img/255

X_train = X_train.reshape(-1, 96,96,1)
Y_train = train.drop('Image', axis=1)

Y_train = Y_train.values

Y_train = np.array(Y_train, dtype='float')

Y_train.shape, X_train.shape
# keras CNN

# 

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding = 'same', activation = 'relu', input_shape = (96,96,1)))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=32, kernel_size=(3,3), padding = 'same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dropout(0.1))

model.add(Dense(30))
optimizer = RMSprop(lr = 0.001, epsilon = 1e-8)

optimizer1 =Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer1, loss = "mse", metrics = ["accuracy"] )


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

tensorboard = TensorBoard(log_dir = './output')

modelcheckpoint = ModelCheckpoint(filepath='./optimized_model.h5', monitor="val_loss", save_best_only=True, mode="min")

callback_list = [learning_rate_reduction, tensorboard, modelcheckpoint]
model1 = Sequential([Flatten(input_shape=(96,96,1)),

                         Dense(128, activation="relu"),

                         Dropout(0.1),

                         Dense(64, activation="relu"),

                         Dense(30)

                         ])



model1.compile(optimizer='adam', 

              loss='mse',

              metrics=['mae','accuracy'])
batch_size =100

epochs = 50

history = model.fit(X_train, Y_train, epochs = epochs, batch_size=batch_size, callbacks=callback_list, validation_split=0.1, verbose = 2)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
#preparing test data

timag = []

for i in range(0,1783):

    timg = test['Image'][i].split(' ')

    timg = ['0' if x == '' else x for x in timg]

    

    timag.append(timg)
X_test = np.array(timag,dtype = 'float')

X_test = X_test/255

X_test = X_test.reshape(-1,96,96,1)

X_test.shape
opt_model = load_model('./optimized_model.h5')

pred = model.predict(X_test, batch_size = 100)

pred.shape
lookid_list = list(look_id['FeatureName'])

imageID = list(look_id['ImageId']-1)

pre_list = list(pred)
rowid = look_id['RowId']

rowid=list(rowid)
feature = []

for f in list( look_id['FeatureName']):

    feature.append(lookid_list.index(f))
preded = []

for x,y in zip(imageID,feature):

    preded.append(pre_list[x][y])
rowid = pd.Series(rowid,name = 'RowId')

loc = pd.Series(preded,name = 'Location')

submission = pd.concat([rowid,loc],axis = 1)

submission.to_csv('face_key_detection_submission.csv',index = False)
!ls