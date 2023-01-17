import numpy as np

from sklearn.metrics import r2_score

from sklearn.cross_validation import train_test_split

import pandas as pd

import subprocess

from scipy.sparse import csr_matrix, hstack

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import KFold

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers.advanced_activations import PReLU
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

sub = pd.read_csv('../input/sample_submission.csv')

y = train['y']

ID = test['ID']

train.drop(['y','ID'],axis=1,inplace= True)

test.drop(['ID'],axis=1,inplace= True)

ntrain = train.shape[0]

xtr_te = pd.concat((train, test), axis = 0)

sparse_data = []

categories = ['X0', 'X1', 'X2', 'X3', 'X4','X5', 'X6', 'X8']

for f in categories:

    dummy = pd.get_dummies(xtr_te[f].astype('category'))

    tmp = csr_matrix(dummy)

    sparse_data.append(tmp)

xtr_te_cat = hstack(sparse_data, format = 'csr')

xtr_te_cat = xtr_te_cat.todense()

drop = []

for i in train.columns:

    if len(train[i].unique()) <2:

        print(len(train[i].unique())), i

        drop.append(i)

xtr_te.drop(categories,axis=1,inplace=True)

xtr_te.drop(drop,axis=1,inplace=True)

xtr_te = np.array(xtr_te)

xtr_te = np.concatenate((xtr_te_cat,xtr_te),axis =1)

xtrain = xtr_te[:ntrain,:]

xtest = xtr_te[ntrain:,:]

from sklearn import preprocessing



xtr_te = np.array(xtr_te)

xtr_te = preprocessing.minmax_scale(xtr_te)

xtrain = xtr_te[:ntrain,:]

xtest = xtr_te[ntrain:,:]
X_train_1,X_test_1,y_train_1,y_test_1 = train_test_split(xtrain,y,test_size=0.4)

X_train_2,X_test_2,y_train_2,y_test_2 = train_test_split(xtrain,y,test_size=0.3)

X_train_3,X_test_3,y_train_3,y_test_3 = train_test_split(xtrain,y,test_size=0.3)

X_train_4,X_test_4,y_train_4,y_test_4 = train_test_split(xtrain,y,test_size=0.3)

X_train_5,X_test_5,y_train_5,y_test_5 = train_test_split(xtrain,y,test_size=0.3)

X_train_6,X_test_6,y_train_6,y_test_6 = train_test_split(xtrain,y,test_size=0.3)

# play around with parameters

batch__size = 128 # try 32,64,16

number_of_epochs = 10 #change it to 10000 to find out the best set of weights

den = 125 # try different one
from keras.callbacks import ModelCheckpoint

import h5py

checkpointer = ModelCheckpoint(filepath="weights_model_1.hdf5", verbose=0, save_best_only=True,monitor='val_loss')

def nn_model():

    model = Sequential()

    model.add(Dense(den, input_dim = xtrain.shape[1], init = 'he_normal',activation='tanh'))

    model.add(Dropout(0.5))

    model.add(Dense(1, init = 'he_normal'))

    model.compile(loss = 'mse', optimizer = 'adadelta')

    return(model)

model_1 = nn_model()

model_1.fit(X_train_1, y_train_1, nb_epoch=number_of_epochs,verbose=0,batch_size=batch__size,

                 callbacks=[checkpointer],validation_data=(X_test_1,y_test_1))



v1 = model_1.evaluate(X_test_1,y_test_1)

print (v1,v1**0.5)



from keras.callbacks import ModelCheckpoint

import h5py

checkpointer = ModelCheckpoint(filepath="weights_model_2.hdf5", verbose=0, save_best_only=True,monitor='val_loss')

def nn_model():

    model = Sequential()

    model.add(Dense(den, input_dim = xtrain.shape[1], init = 'he_normal',activation='tanh'))

    model.add(Dropout(0.5))

    model.add(Dense(1, init = 'he_normal'))

    model.compile(loss = 'mse', optimizer = 'adadelta')

    return(model)

model_2 = nn_model()

model_2.fit(X_train_2, y_train_2, nb_epoch=number_of_epochs,verbose=0,batch_size=batch__size,

                 callbacks=[checkpointer],validation_data=(X_test_2,y_test_2))



v2 = model_2.evaluate(X_test_2,y_test_2)



print (v2,v2**0.5)



from keras.callbacks import ModelCheckpoint

import h5py

checkpointer = ModelCheckpoint(filepath="weights_model_3.hdf5", verbose=0, save_best_only=True,monitor='val_loss')

def nn_model():

    model = Sequential()

    model.add(Dense(den, input_dim = xtrain.shape[1], init = 'he_normal',activation='tanh'))

    model.add(Dropout(0.5))

    model.add(Dense(1, init = 'he_normal'))

    model.compile(loss = 'mse', optimizer = 'adadelta')

    return(model)

model_3 = nn_model()

model_3.fit(X_train_3, y_train_3, nb_epoch=number_of_epochs,verbose=0,batch_size=batch__size,

                 callbacks=[checkpointer],validation_data=(X_test_3,y_test_3))



v3 = model_3.evaluate(X_test_3,y_test_3)



print (v3,v3**0.5)



from keras.callbacks import ModelCheckpoint

import h5py

checkpointer = ModelCheckpoint(filepath="weights_model_4.hdf5", verbose=0, save_best_only=True,monitor='val_loss')

def nn_model():

    model = Sequential()

    model.add(Dense(den, input_dim = xtrain.shape[1], init = 'he_normal',activation='tanh'))

    model.add(Dropout(0.5))

    model.add(Dense(1, init = 'he_normal'))

    model.compile(loss = 'mse', optimizer = 'adadelta')

    return(model)

model_4 = nn_model()

model_4.fit(X_train_4, y_train_4, nb_epoch=number_of_epochs,verbose=0,batch_size=batch__size,

                 callbacks=[checkpointer],validation_data=(X_test_4,y_test_4))



v4 = model_4.evaluate(X_test_4,y_test_4)

print (v4,v4**0.5)



from keras.callbacks import ModelCheckpoint

import h5py

checkpointer = ModelCheckpoint(filepath="weights_model_5.hdf5", verbose=0, save_best_only=True,monitor='val_loss')

def nn_model():

    model = Sequential()

    model.add(Dense(den, input_dim = xtrain.shape[1], init = 'he_normal',activation='tanh'))

    model.add(Dropout(0.5))

    model.add(Dense(1, init = 'he_normal'))

    model.compile(loss = 'mse', optimizer = 'adadelta')

    return(model)

model_5 = nn_model()

model_5.fit(X_train_5, y_train_5, nb_epoch=number_of_epochs,verbose=0,batch_size=batch__size,

                 callbacks=[checkpointer],validation_data=(X_test_5,y_test_5))



v5 = model_5.evaluate(X_test_5,y_test_5)

print (v5,v5**0.5)



from keras.callbacks import ModelCheckpoint

import h5py

checkpointer = ModelCheckpoint(filepath="weights_model_6.hdf5", verbose=0, save_best_only=True,monitor='val_loss')

def nn_model():

    model = Sequential()

    model.add(Dense(den, input_dim = xtrain.shape[1], init = 'he_normal',activation='tanh'))

    model.add(Dropout(0.5))

    model.add(Dense(1, init = 'he_normal'))

    model.compile(loss = 'mse', optimizer = 'adadelta')

    return(model)

model_6 = nn_model()

model_6.fit(X_train_6, y_train_6, nb_epoch=number_of_epochs,verbose=0,batch_size=batch__size,

                 callbacks=[checkpointer],validation_data=(X_test_6,y_test_6))



v6 = model_6.evaluate(X_test_6,y_test_6)

print (v6,v6**0.5)



print ((v1+v2+v3+v4+v5+v6)/6, 'mse final',(v1**0.5+v2**0.5+v3**0.5+v4**0.5+v5**0.5+v6**0.5)/6,'as rmse final')
model_1.load_weights("weights_model_1.hdf5")

model_2.load_weights("weights_model_2.hdf5")

model_3.load_weights("weights_model_3.hdf5")

model_4.load_weights("weights_model_4.hdf5")

model_5.load_weights("weights_model_5.hdf5")

model_6.load_weights("weights_model_6.hdf5")



r1 = r2_score(y_test_1,model_1.predict(X_test_1))

r2 = r2_score(y_test_2,model_2.predict(X_test_2))

r3 = r2_score(y_test_3,model_3.predict(X_test_3))

r4 = r2_score(y_test_4,model_4.predict(X_test_4))

r5 = r2_score(y_test_5,model_5.predict(X_test_5))

r6 = r2_score(y_test_6,model_6.predict(X_test_6))



print (r1,r2,r3,r4,r5,r6)

print ((r1+r2+r3+r4+r5+r6)/6.0)







pred1 = model_1.predict(xtest)

pred2 = model_2.predict(xtest)

pred3 = model_3.predict(xtest)

pred4 = model_4.predict(xtest)

pred5 = model_5.predict(xtest)

pred6 = model_6.predict(xtest)
pred_final2 = (pred1+pred2+pred3+pred4+pred5+pred6)/6.0

sub['y'] = pred_final2

sub.to_csv('output.csv',index=None)