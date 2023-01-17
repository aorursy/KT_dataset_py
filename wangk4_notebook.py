import pandas as pd

import numpy as np

from numpy import isnan

from pandas import DataFrame

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
#read files 

train = pd.read_csv('../input/train.csv', na_values=-1)

test = pd.read_csv('../input/test.csv', na_values=-1)
print(train.shape, test.shape)
#get the header title

list(train)
train.describe()
#drop the id and target cols

X = train.drop(['id', 'target'], axis=1)

features = X.columns

X = X.values

y = train['target'].values
#get the test ids for results

out = test['id'].to_frame()

out['target'] = 0
#get the test data

test_X = test.drop(['id'],axis=1)

test_X = test_X.values
# there is a error message: 

#ValueError: Input contains NaN, infinity or a value too large for dtype('float32').

# for solving this problem, I changed all NaN to 0

whnan = isnan(X)

X[whnan] = 0

whnan = isnan(test_X)

test_X[whnan] = 0
# train the model with 5-fold cross validation 

kfold = 5  # need to change to 5

skf = StratifiedKFold(n_splits=kfold, random_state=0)

temp = np.zeros(shape=(len(test_X),2))

for train_index, test_index in skf.split(X, y):

	X_train, X_valid = X[train_index], X[test_index]

	y_train, y_valid = y[train_index], y[test_index]



	clf = RandomForestClassifier(max_depth=2, random_state=0)

	clf.fit(X_train, y_train)

	prob = clf.predict_proba(test_X)

	temp = np.add(temp,prob)
#get the results

temp =temp / 5

res = DataFrame(temp[:,1])

out['target'] = res

out.to_csv('sub.csv',index = False, float_format = '%.5f')
for train_index, test_index in skf.split(X, y):

	X_train, X_valid = X[train_index], X[test_index]

	y_train, y_valid = y[train_index], y[test_index]



	clf = RandomForestClassifier(n_estimators=50, max_depth=2, max_features="log2", random_state=0)

	clf.fit(X_train, y_train)

	prob = clf.predict_proba(test_X)

	temp = np.add(temp,prob)
#get the results

temp =temp / 5

res = DataFrame(temp[:,1])

out['target'] = res

out.to_csv('sub.csv',index = False, float_format = '%.5f')
#before do another training, 

#I want to analysis the feature importance of the training data

import matplotlib.pyplot as plt

clf = RandomForestClassifier(n_estimators=50, max_depth=2, max_features="log2", random_state=0)

clf.fit(X,y)

importances = clf.feature_importances_
y_pos = np.arange(len(importances))

plt.figure(figsize=(6,12))

plt.title('Feature Importances')

plt.barh(y_pos, importances, color='b')

plt.yticks(y_pos,features)

plt.xlabel('Relative Importance')

plt.show()
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  

test = test.drop(col_to_drop, axis=1)  
print(train.shape, test.shape)
#without the "ps_calc_*" features model training

X = train.drop(['id', 'target'], axis=1)

features = X.columns

X = X.values

y = train['target'].values

out = test['id'].to_frame()

out['target'] = 0

test_X = test.drop(['id'],axis=1)

test_X = test_X.values
whnan = isnan(X)

X[whnan] = 0

whnan = isnan(test_X)

test_X[whnan] = 0
for train_index, test_index in skf.split(X, y):

	X_train, X_valid = X[train_index], X[test_index]

	y_train, y_valid = y[train_index], y[test_index]



	clf = RandomForestClassifier(n_estimators=50, max_depth=2, max_features="log2", random_state=0)

	clf.fit(X_train, y_train)

	prob = clf.predict_proba(test_X)

	temp = np.add(temp,prob)
#get the results

temp =temp / 5

res = DataFrame(temp[:,1])

out['target'] = res

out.to_csv('/home/kaiwang/Desktop/project1/sub.csv',index = False, float_format = '%.5f')
#try to use multiple layer perceptron

#load packages



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from sklearn.model_selection import train_test_split
#split the data into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#define the model

model = Sequential()

model.add(Dense(512, input_dim=37, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))
#compile the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model

model.fit(X_train, y_train, epochs=3, batch_size=124)
#evaluate the model

scores = model.evaluate(X_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#predict the test reuslts

pres = model.predict_proba(test_X,verbose = 0)

res = DataFrame(pres)

out['target'] = res

out.to_csv('sub.csv', index=False, float_format='%.5f') 
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

two_way = [('ps_car_13', 'ps_ind_17_bin'), ('ps_car_13', 'ps_ind_05_cat'),

           ('ps_ind_05_cat', 'ps_reg_03'), ('ps_ind_05_cat', 'ps_ind_17_bin'),

           ('ps_car_13', 'ps_reg_03'), ('ps_ind_17_bin', 'ps_reg_03')]



three_way = [('ps_car_13', 'ps_ind_05_cat', 'ps_ind_17_bin'),

             ('ps_car_13', 'ps_ind_05_cat', 'ps_reg_03'),

             ('ps_car_13', 'ps_car_13', 'ps_ind_17_bin'),

             ('ps_ind_05_cat', 'ps_ind_17_bin', 'ps_reg_03'),

             ('ps_car_13', 'ps_ind_17_bin', 'ps_reg_03'),

             ('ps_car_13', 'ps_ind_04_cat', 'ps_ind_17_bin')]
for (x1, x2) in two_way:

  train[x1 + '_' + x2] = train[x1] * train[x2]

  test[x1 + '_' + x2] = test[x1] * test[x2]



for (x1, x2, x3) in three_way:

  train[x1 + '_' + x2 + '_' + x3] = train[x1] * train[x2] * train[x3]

  test[x1 + '_' + x2 + '_' + x3] = test[x1] * test[x2] * test[x3]
print('Train shape:', train.shape)

print('Test shape:', test.shape)



print('Columns:', train.columns)
y_train = train['target'].values

id_train = train['id'].values

id_test = test['id'].values



train = train.drop(['target', 'id'], axis=1)

test = test.drop(['id'], axis=1)
x_train = np.array(train)

x_test = np.array(test)
print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)
#use cross validation for multiple layer perceptron training

from keras.optimizers import Adam

from keras.utils import np_utils

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

from keras.layers.normalization import BatchNormalization



skf = StratifiedKFold(n_splits=5, random_state=0)



ntest = len(x_test)



oof_test = np.zeros((ntest,))

oof_test_kf = np.empty((5, ntest))



for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):

    print(i, ' of ', 5-1)

    x_tr = x_train[train_index]

    x_te = x_train[test_index]

    

    y_tr = np_utils.to_categorical(y_train[train_index])

    y_te = np_utils.to_categorical(y_train[test_index])

    

    kfold_weight_path = 'nn' + str(i) + '.h5'

    

    model = Sequential()

    model.add(Dense(518, activation = 'relu', input_shape=(x_tr.shape[1],)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Dense(256, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Dense(128, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Dense(2, activation='sigmoid'))

    

    model.compile(loss = 'binary_crossentropy',

                  optimizer = Adam(),

                  metrics=['accuracy'])

    callback = [EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1),

                ModelCheckpoint(kfold_weight_path, monitor='val_loss', save_best_only=True, verbose=0),

                ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 6, verbose = 1)]

    model.fit(x_tr, y_tr,

              batch_size=1024*2, 

              epochs=10000,

              verbose=1,

              validation_data=(x_te, y_te), 

              callbacks = callback)

    oof_test_kf[i, :] = model.predict(x_test)[:, 1]
#get the results

oof_test[:] = oof_test_kf.mean(axis=0)

pred = oof_test.reshape(-1, 1).ravel()
sub = pd.DataFrame()

sub['id'] = id_test

sub['target'] = pred

sub.to_csv('sub.csv', index=False, float_format='%.5f') 
import xgboost as xgb

x_train = np.array(train)

x_test = np.array(test)



d_test = xgb.DMatrix(np.array(x_test))
#train the model

for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):

    print(i, ' of ', 5-1)

    x_tr = x_train[train_index]

    x_te = x_train[test_index]

    

    y_tr = y_train[train_index]

    y_te = y_train[test_index]

    

    ratio = float(np.sum(y_tr == 1)) / np.sum(y_tr == 0)

    

    dtra = xgb.DMatrix(data = x_tr, label = y_tr)

    dval = xgb.DMatrix(data = x_te, label = y_te) 

    

    watchlist  = [(dtra,'train'), (dval,'eval')]



    xgb_params = {

        'min_child_weight': 4,

        'eval_metric': 'auc',

        'eta': 0.0125,

        'colsample_bytree': 0.8,

        'max_depth': 12,

        'subsample': 0.8,

        'alpha': 1,

        'gamma': 1,

        'silent': 1,

        'seed': 0,

        'nthread':-1,

        'n_parallel_tree': 1

    }



    xgb_mod = xgb.train(xgb_params, 

                        dtra,

                        10000, 

                        watchlist, 

                        early_stopping_rounds=100, 

                        maximize=True, 

                        verbose_eval=100)

    oof_test_kf[i, :] = xgb_mod.predict(d_test, ntree_limit=xgb_mod.best_ntree_limit+50)

    
#get the results

oof_test[:] = oof_test_kf.mean(axis=0)

pred = oof_test.reshape(-1, 1).ravel()



sub = pd.DataFrame()

sub['id'] = id_test

sub['target'] = pred

sub.to_csv('sub.csv', index=False, float_format='%.5f') 