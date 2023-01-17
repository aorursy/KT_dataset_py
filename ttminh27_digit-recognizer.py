import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator
# init random seed

np.random.seed(1)



# Load the data from csv to dataframe

X_raw = pd.read_csv("../input/digit-recognizer/train.csv")

X_test_raw = pd.read_csv("../input/digit-recognizer/test.csv")



y = X_raw["label"] #ground truth

X = X_raw.drop(labels = ["label"],axis = 1) 



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=1)



WIDTH=28

HEIGHT=28

NUM_CLASSES=10



X_train = X_train.values

y_train = y_train.values



X_valid = X_valid.values

y_valid = y_valid.values



# create one-hot vector of the ground truth

y_train_oh = to_categorical(y_train, num_classes = NUM_CLASSES)

y_valid_oh = to_categorical(y_valid, num_classes = NUM_CLASSES)



X_test = X_test_raw.values
plt.subplot(1,2,1)

sns.countplot(y_train).set_title('y_train')

plt.subplot(1,2,2)

sns.countplot(y_valid).set_title('y_valid')

plt.tight_layout()

print('Missing value X_raw: ' + str(X_raw.isnull().values.any()))
def showImg(X,y_hat,y,n_row=4, n_col=4):

    plt.figure(figsize=(1.5*n_row,1.5*n_col))

    for i in range(n_row*n_col):

        plt.subplot(n_row,n_col,i+1)

        plt.tight_layout()

        plt.imshow(X[i].reshape(WIDTH,HEIGHT),cmap='cool')

        plt.title('{} - {}'.format(y_hat[i],y[i]))

        plt.axis('off')

        

showImg(X_train,y_train,y_train,8,8)
X_train.max()
def reshape_channel(X):

    return X.reshape(-1,HEIGHT,WIDTH,1)



def preprocessData(X):

    X = reshape_channel(X)

    return X
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam



optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.9999, amsgrad=True)



model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'same',activation ='relu',input_shape = (HEIGHT,WIDTH,1)))

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'same',activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'same',activation ='relu'))

model.add(MaxPool2D(pool_size=(5,5),padding='same'))

model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))

model.add(MaxPool2D(pool_size=(3,3),padding='same'))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.2))

model.add(Dense(NUM_CLASSES, activation = "softmax"))

# model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

model.compile(optimizer ='adam', loss = "categorical_crossentropy", metrics=["accuracy"])

model.summary()
import math

from keras.callbacks import LearningRateScheduler





n_epoch=60

drop=0.6

epoch_drop = 5

initial_lrate = 0.001

min_lrate = 0.00001



def step_decay(epoch):

    lrate = initial_lrate*math.pow(drop,math.floor((1+epoch)/epoch_drop))

    if(lrate<min_lrate):

        lrate=min_lrate

    

    print('learning rate at epoch ' +str(epoch+1) +': ' + str(lrate))

    return lrate



lrate = LearningRateScheduler(step_decay)
from keras.preprocessing.image import ImageDataGenerator

# create data generator

datagen_with_aug = ImageDataGenerator(

    rotation_range=10,

    zoom_range = 0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    rescale=1./255,

)



datagen_with_aug.fit(preprocessData(X_train))



datagen_no_aug = ImageDataGenerator()

# create iterator



train_generator = datagen_with_aug.flow(preprocessData(X_train), y_train_oh, batch_size=128)

# valid_generator = datagen_no_aug.flow(preprocessData(X_valid), y_valid_oh)

valid_generator = datagen_with_aug.flow(preprocessData(X_valid), y_valid_oh,batch_size=128)
X_sample = X_train[2:3,]

y_sample = y_train[:1,]

size=6

plt.figure(figsize=(1.5*size,1.5*size))

plt.subplot(size,size,1)

plt.imshow(X_sample.reshape(HEIGHT,WIDTH),cmap='copper')

plt.title('ORIGIN')

plt.axis('off')

for i in range(1,size*size):

    train_draw=datagen_with_aug.flow(preprocessData(X_sample),y_sample,batch_size=1)

    X_t,_=train_draw.next()

    plt.subplot(size,size,i+1)

    plt.imshow(X_t[0].reshape(HEIGHT,WIDTH),cmap='cool')

    plt.title('augmented')

    plt.axis('off')
import keras



keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1)



hist = model.fit_generator(train_generator,validation_data=valid_generator,callbacks=[lrate],epochs=n_epoch,steps_per_epoch=(X_train.shape[0]/128)*2)

# hist = model.fit(preprocessData(X_train),y_train_oh,validation_data=(preprocessData(X_valid),y_valid_oh),callbacks=[lrate],epochs=n_epoch)

# hist = model.fit(preprocessData(X_train),y_train_oh,validation_data=(preprocessData(X_valid),y_valid_oh),epochs=50)
y_pred = model.predict(preprocessData(X_test))

y_pred = np.argmax(y_pred,axis = 1)

submission = pd.DataFrame({'ImageId':range(1,28001),'Label':y_pred})

submission.to_csv('submission.csv',index=False)
# X_test_raw

y_valid_pred = model.predict(preprocessData(X_valid))

y_valid_pred = np.argmax(y_valid_pred,axis = 1)

showImg(X_valid,y_valid_pred,y_valid,8,8)
print(hist.history.keys())

fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(15,30))



ax1.plot(np.arange(1,len(hist.history['accuracy'])+1,1),hist.history['accuracy'],label='train accuracy')

ax1.plot(np.arange(1,len(hist.history['val_accuracy'])+1,1),hist.history['val_accuracy'],label='validation accuracy')

ax1.grid()

ax1.legend()



ax2.plot(np.arange(1,len(hist.history['loss'])+1,1),hist.history['loss'],label='train loss')

ax2.plot(np.arange(1,len(hist.history['val_loss'])+1,1),hist.history['val_loss'],label='validation loss')

ax2.grid()

ax2.legend()



ax3.plot(np.arange(1,len(hist.history['lr'])+1,1),hist.history['lr'],label='learning rate')

ax3.grid()

ax3.legend()