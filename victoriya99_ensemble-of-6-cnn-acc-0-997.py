#Basic 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')



np.random.seed(9)
#Specific



import matplotlib.image as mpimg

from sklearn.metrics import confusion_matrix

import itertools



import tensorflow as tf



from keras.utils.np_utils import to_categorical #convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import LearningRateScheduler

#GPU testing 



if tf.test.gpu_device_name():

    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:

    print("Please install GPU version of TF")

print(tf.test.is_built_with_cuda())
train=pd.read_csv('../input/digit-recognizer/train.csv')

test=pd.read_csv('../input/digit-recognizer/test.csv')
y_train=train['label']

X_train=train.drop(columns='label')
# free some space

del train 
sns.countplot(y_train)
X_train.isna().any().sum()
test.isna().any().sum()
X_train /=255.0

test /=255.0
X_train=X_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)
#Cross-entropy loss needs OHE for target

y_train=to_categorical(y_train,num_classes=10)
#X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size = 0.25, random_state=2)
plt.imshow(X_train[0][:,:,0])
X_train.shape
nets=6

model=[0]*nets

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) 

#Sequential model

model[0]=Sequential()



# 2 convolutional layers with 32 filters, 1 subsampling layer, 1 regularization layer

model[0].add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))

model[0].add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))

model[0].add(MaxPool2D(pool_size=(2,2)))

model[0].add(Dropout(0.25))



# 2 convolutional layers with 64 filters, 1 subsampling layer, 1 regularization layer

model[0].add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu',input_shape=(28,28,1)))

model[0].add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu',input_shape=(28,28,1)))

model[0].add(MaxPool2D(pool_size=(2,2)))

model[0].add(Dropout(0.25))



#Classification layers

model[0].add(Flatten())  #Full conected layer needs 1D array on input

model[0].add(Dense(256,activation='relu'))

model[0].add(Dropout(0.5))

model[0].add(Dense(10,activation='softmax'))



model[0].compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'] )
model[1]=Sequential()



# 2 convolutional layers with 32 filters, 1 subsampling layer, 1 regularization layer

model[1].add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))

model[1].add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))

model[1].add(MaxPool2D(pool_size=(2,2)))

model[1].add(Dropout(0.25))



# 2 convolutional layers with 64 filters, 1 subsampling layer, 1 regularization layer

model[1].add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu',input_shape=(28,28,1)))

model[1].add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu',input_shape=(28,28,1)))

model[1].add(MaxPool2D(pool_size=(2,2)))

model[1].add(Dropout(0.25))



#Classification layers

model[1].add(Flatten())  #Full conected layer needs 1D array on input

model[1].add(Dense(256,activation='relu'))

model[1].add(Dropout(0.5))

model[1].add(Dense(10,activation='softmax'))



model[1].compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'] )
#Sequential model

model[2]=Sequential()



# 2 convolutional layers, BatchNormalization, 1 subsampling layer, 1 regularization layer

model[2].add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))

model[2].add(BatchNormalization())

model[2].add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))

model[2].add(BatchNormalization())

model[2].add(MaxPool2D(pool_size=(2,2)))

model[2].add(Dropout(0.25))



# 2 convolutional layers with 64 filters, BatchNormalization, 1 subsampling layer, 1 regularization layer

model[2].add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu',input_shape=(28,28,1)))

model[2].add(BatchNormalization())

model[2].add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu',input_shape=(28,28,1)))

model[2].add(BatchNormalization())

model[2].add(MaxPool2D(pool_size=(2,2)))

model[2].add(Dropout(0.25))



#Classification layers

model[2].add(Flatten())  #Full conected layer needs 1D array on input

model[2].add(Dense(256,activation='relu'))

model[2].add(BatchNormalization())

model[2].add(Dropout(0.5))

model[2].add(Dense(10,activation='softmax'))



model[2].compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'] )
#Sequential model

model[3]=Sequential()



# 2 convolutional layers, BatchNormalization, 1 subsampling layer, 1 regularization layer

model[3].add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))

model[3].add(BatchNormalization())

model[3].add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))

model[3].add(BatchNormalization())

model[3].add(MaxPool2D(pool_size=(2,2)))

model[3].add(Dropout(0.25))



# 2 convolutional layers with 64 filters, BatchNormalization, 1 subsampling layer, 1 regularization layer

model[3].add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu',input_shape=(28,28,1)))

model[3].add(BatchNormalization())

model[3].add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu',input_shape=(28,28,1)))

model[3].add(BatchNormalization())

model[3].add(MaxPool2D(pool_size=(2,2)))

model[3].add(Dropout(0.25))



#Classification layers

model[3].add(Flatten())  #Full conected layer needs 1D array on input

model[3].add(Dense(256,activation='relu'))

model[3].add(BatchNormalization())

model[3].add(Dropout(0.5))

model[3].add(Dense(10,activation='softmax'))



model[3].compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'] )
#Sequential model

model[4]=Sequential()



model[4].add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))

model[4].add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

model[4].add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

model[4].add(MaxPool2D(pool_size=2))

model[4].add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

model[4].add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))

model[4].add(MaxPool2D(pool_size=2))

model[4].add(Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'))

model[4].add(MaxPool2D(pool_size=2, padding='same'))

model[4].add(Flatten())

model[4].add(Dense(256, activation='relu'))

model[4].add(Dense(10, activation='softmax'))



model[4].compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'] )
#Sequential model

model[5]=Sequential()



model[5].add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))

model[5].add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

model[5].add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

model[5].add(MaxPool2D(pool_size=2))

model[5].add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

model[5].add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))

model[5].add(MaxPool2D(pool_size=2))

model[5].add(Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'))

model[5].add(MaxPool2D(pool_size=2, padding='same'))

model[5].add(Flatten())

model[5].add(Dense(256, activation='relu'))

model[5].add(Dense(10, activation='softmax'))



model[5].compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'] )
# With data augmentation to prevent overfitting:

datagen = ImageDataGenerator(

          featurewise_center=False,            # set input mean to 0 over the dataset

          samplewise_center=False,             # set each sample mean to 0

          featurewise_std_normalization=False, # divide inputs by std of the dataset

          samplewise_std_normalization=False,  # divide each input by its std

          zca_whitening=False,                 # apply ZCA whitening

          rotation_range=10,                   # randomly rotate images in the range (degrees, 0 to 180)

          zoom_range = 0.1,                    # Randomly zoom image 

          width_shift_range=0.1,               # randomly shift images horizontally (fraction of total width)

          height_shift_range=0.1,              # randomly shift images vertically (fraction of total height)

          horizontal_flip=False,               # randomly flip images

          vertical_flip=False)                 # randomly flip images



datagen.fit(X_train)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history = [0] * nets
epochs = 20

for j in range(nets):

    print("CNN ",j+1)

    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, y_train, test_size = 0.1)

    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=32),

        epochs = epochs, steps_per_epoch = X_train2.shape[0]//32,  

        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=1)

    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

        j+1,epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
data={'CNN1':[history[0].history['val_accuracy'][19]],'CNN2':[history[1].history['val_accuracy'][19]],'CNN3':[history[2].history['val_accuracy'][19]],'CNN4':[history[3].history['val_accuracy'][19]],'CNN5':[history[4].history['val_accuracy'][19]],'CNN6':[history[5].history['val_accuracy'][19]]}



histories=pd.Series(data)

                                

histories
#make the ensemble prediction on the validation set

results_ens = np.zeros( (X_val2.shape[0],10) ) 

for j in range(nets):

    results_ens = results_ens + model[j].predict(X_val2)

results_ens = np.argmax(results_ens,axis =1)
#get the score of the ensemble prediction

from sklearn.metrics import accuracy_score



Y_val2=np.argmax(Y_val2,axis =1)

ens_acc=accuracy_score(results_ens,Y_val2)

print('Ensemble accuracy: {:.4f}'.format (ens_acc))
fig, ax = plt.subplots(2,1)

ax[0].plot(history[5].history['loss'], color='b', label="Training loss")

ax[0].plot(history[5].history['val_loss'], color='r', label="Validation loss",axes =ax[0])

ax[0].grid(color='black', linestyle='-', linewidth=0.25)

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history[5].history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history[5].history['val_accuracy'], color='r',label="Validation accuracy")

ax[1].grid(color='black', linestyle='-', linewidth=0.25)

legend = ax[1].legend(loc='best', shadow=True)
#Vizualization of single CNNs and the ensemble accuracy on validation set 

accs=[]

for i in range (nets):

    accs.append(history[i].history['val_accuracy'][19])

                 

accs.append(ens_acc)

cnns=['CNN1','CNN2','CNN3','CNN4','CNN5','CNN6','ENS']



fig,ax=plt.subplots(figsize=(8,6))

ax=sns.barplot(x=cnns,y=accs)

ax.set(ylim=(0.98, 1.0))



plt.show()
#predict the labels of the test 

results = np.zeros( (test.shape[0],10) ) 

for j in range(nets):

    results = results + model[j].predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("predictions.csv",index=False)