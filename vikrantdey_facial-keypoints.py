import numpy as np 

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
print("Contents of input/facial-keypoints-detection directory: ")

!ls ../input/facial-keypoints-detection/



print("\nExtracting .zip dataset files to working directory ...")

!unzip -u ../input/facial-keypoints-detection/test.zip

!unzip -u ../input/facial-keypoints-detection/training.zip



print("\nCurrent working directory:")

!pwd

print("\nContents of working directory:")

!ls
from sklearn.model_selection import train_test_split 

from matplotlib import pyplot as plt

%matplotlib inline 
IdLookupTable = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')

IdLookupTable.info()
IdLookupTable.head()
training = pd.read_csv('/kaggle/working/training.csv')

training.info()
training.head(5)
test = pd.read_csv('/kaggle/working/test.csv')

test.info()
test.head(5)
training = training.dropna()

training.shape, type(training)
training['Image'] = training['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))
def get_image_and_dots(df, index):

    image = plt.imshow(df['Image'][index],cmap='gray')

    l = []

    for i in range(1,31,2):

        l.append(plt.plot(df.loc[index][i-1], df.loc[index][i], 'ro'))

        

    return image, l
fig = plt.figure(figsize=(8, 8))

fig.subplots_adjust(

    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)



for i in range(16):

    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])

    get_image_and_dots(training, i)



plt.show()

plt.savefig('fkeypoints.png')
X = np.asarray([training['Image']], dtype=np.uint8).reshape(training.shape[0],96,96,1)

y = training.drop(['Image'], axis=1)

X.shape, y.shape, type(X), type(y)
Y = y.to_numpy()

type(Y), Y.shape
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, AvgPool2D, BatchNormalization, Dropout, Activation, MaxPooling2D

from keras.optimizers import Adam

from keras import regularizers

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential, Model

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
model = Sequential()



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

# model.add(BatchNormalization())

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())





model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(30))

model.summary()
model.compile(optimizer='Adam', 

              loss='mse', 

              metrics=['mae'])



model.fit(X_train, y_train, epochs=600)
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
model.save('keypoint_model03.h5')
test['Image'] = test['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))

test.shape, type(test)
test_X = np.asarray([test['Image']], dtype=np.uint8).reshape(test.shape[0],96,96,1)

test_res = model.predict(test_X)



train_predicts = model.predict(X_train)
n = 12



xv = X_train[n].reshape((96,96))

plt.imshow(xv,cmap='gray')



for i in range(1,31,2):

    plt.plot(train_predicts[n][i-1], train_predicts[n][i], 'ro')

    plt.plot(y_train[n][i-1], y_train[n][i], 'x', color='green')



plt.show()
lookid_dir = '../input/facial-keypoints-detection/IdLookupTable.csv'

lookid_data = pd.read_csv(lookid_dir)
lookid_list = list(lookid_data['FeatureName'])

imageID = list(lookid_data['ImageId']-1)

pre_list = list(test_res)

rowid = lookid_data['RowId']

rowid=list(rowid)

feature = []

for f in list(lookid_data['FeatureName']):

    feature.append(lookid_list.index(f))

preded = []

for x,y in zip(imageID,feature):

    preded.append(pre_list[x][y])

rowid = pd.Series(rowid,name = 'RowId')

loc = pd.Series(preded,name = 'Location')

submission = pd.concat([rowid,loc],axis = 1)

submission.to_csv('face_key_detection_submission.csv',index = False)