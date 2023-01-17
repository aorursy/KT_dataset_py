## Measure execution time, becaus Kaggle cloud fluctuates  

import time

start = time.time()



## Importing standard libraries

%pylab inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



## Importing sklearn libraries

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedShuffleSplit



## Keras Libraries for Neural Networks

from keras.models import Sequential

from keras.layers import Merge

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers.advanced_activations import PReLU

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping



## Read data from the CSV file

data = pd.read_csv('../input/train.csv')

parent_data = data.copy()    ## Always a good idea to keep a copy of original data

ID = data.pop('id')



data.shape

data.describe()



## Since the labels are textual, so we encode them categorically

y = data.pop('species')

y = LabelEncoder().fit(y).transform(y)

print(y.shape)



## Most of the learning algorithms are prone to feature scaling

## Standardising the data to give zero mean =)

from sklearn import preprocessing

X = preprocessing.MinMaxScaler().fit(data).transform(data)

X = StandardScaler().fit(data).transform(data)

## normalizing does not help here; l1 and l2 allowed

## X = preprocessing.normalize(data, norm='l1')

print(X.shape)

X



## We will be working with categorical crossentropy function

## It is required to further convert the labels into "one-hot" representation

y_cat = to_categorical(y)

print(y_cat.shape)



## retain class balances 

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1,random_state=12345)

train_index, val_index = next(iter(sss.split(X, y)))

x_train, x_val = X[train_index], X[val_index]

y_train, y_val = y_cat[train_index], y_cat[val_index]

print("x_train dim: ",x_train.shape)

print("x_val dim:   ",x_val.shape)

print()



## --------------------------------------------------

## Developing a layered model for Neural Networks No/1/

## Input dimensions should be equal to the number of features

## We used softmax layer to predict a uniform probabilistic distribution of outcomes

model1 = Sequential()

model1.add(Dense(600,input_dim=192,  init='uniform', activation='relu'))

model1.add(Dropout(0.3))

model1.add(Dense(600, activation='sigmoid'))

model1.add(Dropout(0.3))

model1.add(Dense(99, activation='softmax'))



## Error is measured as categorical crossentropy or multiclass logloss

## Adagrad, rmsprop, SGD, Adadelta, Adam, Adamax, Nadam

model1.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])



## Fitting the model on the whole training data

early_stopping = EarlyStopping(monitor='val_loss', patience=300)

history = model1.fit(x_train, y_train,batch_size=192,nb_epoch=2500 ,verbose=0,

                    validation_data=(x_val, y_val),callbacks=[early_stopping])

                    

## we need to consider the loss for final submission to leaderboard

print('val_acc: ',max(history.history['val_acc']))

print('val_loss: ',min(history.history['val_loss']))

print('train_acc: ',max(history.history['acc']))

print('train_loss: ',min(history.history['loss']))

print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))



## summarize history for loss

## Plotting the loss with the number of iterations

plt.semilogy(history.history['loss'])

plt.semilogy(history.history['val_loss'])

plt.title('model1 loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



## Plotting the error with the number of iterations

## With each iteration the error reduces smoothly

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model1 accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



## --------------------------------------------------

## Developing a layered model for Neural Networks No/2/

## Input dimensions should be equal to the number of features

## We used softmax layer to predict a uniform probabilistic distribution of outcomes

model2 = Sequential()

model2.add(Dense(1024,input_dim=192,  init='glorot_normal', activation='relu'))

model2.add(Dropout(0.2))

model2.add(Dense(512, activation='sigmoid'))

model2.add(Dropout(0.2))

model2.add(Dense(99, activation='softmax'))



## Error is measured as categorical crossentropy or multiclass logloss

## Adagrad, rmsprop, SGD, Adadelta, Adam, Adamax, Nadam

model2.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])



## Fitting the model on the whole training data

early_stopping = EarlyStopping(monitor='val_loss', patience=300)

history = model2.fit(x_train, y_train,batch_size=192,nb_epoch=2500 ,verbose=0,

                    validation_data=(x_val, y_val),callbacks=[early_stopping])

                    

## we need to consider the loss for final submission to leaderboard

print('val_acc: ',max(history.history['val_acc']))

print('val_loss: ',min(history.history['val_loss']))

print('train_acc: ',max(history.history['acc']))

print('train_loss: ',min(history.history['loss']))

print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))



## summarize history for loss

## Plotting the loss with the number of iterations

plt.semilogy(history.history['loss'])

plt.semilogy(history.history['val_loss'])

plt.title('model2 loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



## Plotting the error with the number of iterations

## With each iteration the error reduces smoothly

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model2 accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



## --------------------------------------------------

## Developing a layered model for Neural Networks No/3/

## Input dimensions should be equal to the number of features

## We used softmax layer to predict a uniform probabilistic distribution of outcomes

model3 = Sequential()

model3.add(Dense(600,input_dim=192,  init='uniform', activation='relu'))

model3.add(Dropout(0.3))

model3.add(Dense(600, activation='sigmoid'))

model3.add(Dropout(0.3))

model3.add(Dense(99, activation='softmax'))



## Error is measured as categorical crossentropy or multiclass logloss

## Adagrad, rmsprop, SGD, Adadelta, Adam, Adamax, Nadam

model3.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])



## Fitting the model on the whole training data

early_stopping = EarlyStopping(monitor='val_loss', patience=300)

history = model3.fit(x_train, y_train,batch_size=192,nb_epoch=2500 ,verbose=0,

                    validation_data=(x_val, y_val),callbacks=[early_stopping])

                    

## we need to consider the loss for final submission to leaderboard

print('val_acc: ',max(history.history['val_acc']))

print('val_loss: ',min(history.history['val_loss']))

print('train_acc: ',max(history.history['acc']))

print('train_loss: ',min(history.history['loss']))

print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))



## summarize history for loss

## Plotting the loss with the number of iterations

plt.semilogy(history.history['loss'])

plt.semilogy(history.history['val_loss'])

plt.title('model3 loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



## Plotting the error with the number of iterations

## With each iteration the error reduces smoothly

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model3 accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



## --------------------------------------------------

## Developing a layered model for Neural Networks No/4/

## Input dimensions should be equal to the number of features

## We used softmax layer to predict a uniform probabilistic distribution of outcomes

model4 = Sequential()

model4.add(Dense(768,input_dim=192,  init='uniform', activation='relu'))

model4.add(Dropout(0.3))

model3.add(Dense(384, activation='sigmoid'))

model3.add(Dropout(0.3))

model4.add(Dense(99, activation='softmax'))



## Error is measured as categorical crossentropy or multiclass logloss

## Adagrad, rmsprop, SGD, Adadelta, Adam, Adamax, Nadam

model4.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])



## Fitting the model on the whole training data

early_stopping = EarlyStopping(monitor='val_loss', patience=600)

history = model4.fit(x_train, y_train,batch_size=192,nb_epoch=2500 ,verbose=0,

                    validation_data=(x_val, y_val),callbacks=[early_stopping])

                    

## we need to consider the loss for final submission to leaderboard

print('val_acc: ',max(history.history['val_acc']))

print('val_loss: ',min(history.history['val_loss']))

print('train_acc: ',max(history.history['acc']))

print('train_loss: ',min(history.history['loss']))

print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))



## summarize history for loss

## Plotting the loss with the number of iterations

plt.semilogy(history.history['loss'])

plt.semilogy(history.history['val_loss'])

plt.title('model4 loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



## Plotting the error with the number of iterations

## With each iteration the error reduces smoothly

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model4 accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



#----------------------------------------------

#----------------------------------------------

## read test file

test = pd.read_csv('../input/test.csv')

index = test.pop('id')



## we need to perform the same transformations from the training set to the test set

test = preprocessing.MinMaxScaler().fit(test).transform(test)

test = StandardScaler().fit(test).transform(test)

yPred1 = model1.predict_proba(test)

yPred2 = model2.predict_proba(test)

yPred3 = model3.predict_proba(test)

yPred4 = model4.predict_proba(test)



## average all models

yPred = (yPred1 + yPred2 + yPred3 + yPred4 ) / 4.0



## Converting the test predictions in a dataframe as depicted by sample submission

yPred = pd.DataFrame(yPred,index=index,columns=sort(parent_data.species.unique()))



# show data frame

yPred



## write submission to file

fp = open('submission_nn_kernel.csv','w')

fp.write(yPred.to_csv())



## print run time

end = time.time()

print()

print(round((end-start),2), "seconds")