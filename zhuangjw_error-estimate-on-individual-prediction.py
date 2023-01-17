%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats

import sklearn
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
ds = xr.open_mfdataset('../input/*.nc4').drop('KPP_RFactive')
ds.dims
df = ds.isel(time=[0,1], drop=True).to_dataframe().reset_index(drop=True).rename(columns=lambda s: s.replace('KPP_', ''))

df_y = df.iloc[:, df.columns.str.startswith('AFTER_CHEM')].rename(columns=lambda s: s.replace('AFTER_CHEM_', ''))

df_x = df.iloc[:, df.columns.str.startswith('BEFORE_CHEM')].rename(columns=lambda s: s.replace('BEFORE_CHEM_', ''))

df_in = df.iloc[:, ~df.columns.str.startswith('AFTER_CHEM')].rename(columns=lambda s: s.replace('BEFORE_CHEM_', ''))
X_all= StandardScaler().fit_transform(df_in)

# predict the difference
# pick up a single variable for now
Y_all = StandardScaler().fit_transform( (df_y-df_x)[['O3']]) 

X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)
X_train.shape
def train_member():
    '''train one ensemble member'''
    model = Sequential([
        Dense(100, input_shape=(127,)),
        Activation('relu'),
        Dense(10),
        Activation('relu'),
        Dense(1),
    ])

    model.compile(optimizer='adam',
                  loss='mse')

    model.fit(X_train, Y_train, epochs=5, batch_size=128)
    
    return model
%%time
n_fold = 5 # number of ensemble members

models = []
for _ in range(n_fold):
    models.append(train_member())
# place holder for ensemble prediction
Y_train_preds = np.empty([n_fold, *Y_train.shape])
Y_train_preds.shape
%%time
for i in range(n_fold):
    print('.', end='')
    Y_train_preds[i] = (models[i].predict(X_train))
for Y_train_pred in Y_train_preds:
    print(r2_score(Y_train_pred, Y_train))
# place holder for ensemble prediction
Y_test_preds = np.empty([n_fold, *Y_test.shape])
Y_test_preds.shape
%%time
for i in range(n_fold):
    print('.', end='')
    Y_test_preds[i] = (models[i].predict(X_test))
for Y_test_pred in Y_test_preds:
    print(r2_score(Y_test_pred, Y_test))
Y_test_pred_mean = Y_test_preds.mean(axis=0)
r2_score(Y_test_pred_mean, Y_test)
# too many points
plt.plot([-10, 15], [-10, 15], c='k')
plt.scatter(Y_test, Y_test_pred_mean, alpha=0.2, s=4)

plt.xlim(-10, 15); plt.ylim(-10, 15)
plt.hexbin(Y_test, Y_test_pred_mean, bins='log')
plt.colorbar()
plt.plot([-10, 15], [-10, 15], c='white')
plt.xlim(-10, 15)
plt.ylim(-10, 15)
Y_test_pred_std = Y_test_preds.std(axis=0)
Y_test_pred_std.shape
true_error = np.abs(Y_test - Y_test_pred_mean)
plt.scatter(Y_test_pred_std, true_error, alpha=0.1)
plt.xlabel('ensemble variance')
plt.ylabel('true error')
# data points too condensed, can't see clearly
plt.hexbin(Y_test_pred_std, true_error, bins='log')
plt.xlabel('ensemble variance')
plt.ylabel('true error')
# data points too condensed, can't see clearly
stats.pearsonr(Y_test_pred_std, true_error)
stats.spearmanr(Y_test_pred_std, true_error)
# histogram of error distribution in log scale. Outliers are very few!
plt.hist(true_error, log=True);
true_error.mean() # average error is very small!
true_error.max() # maximum error is very large!
for thres in [30, 20, 10, 5, 2, 1, 0.5, 0.2]:
    mask = (true_error > thres)
    print(thres, ':',mask.sum(), ';', mask.mean())
    
# only 1% samples have error larger than 0.5
# histogram of ensemble variance
# not as skewed as true error distribution, but also relatively few outliers
plt.hist(Y_test_pred_std, log=True);
for thres in [4, 3, 2, 1, 0.5, 0.2]:
    mask = (Y_test_pred_std > thres)
    print(thres, ':',mask.sum(), ';', mask.mean())
    
# only <2% samples have error larger than 0.5
for thres in [4, 3, 2, 1, 0.5, 0.2, 0.1]:
    error_filtered = true_error[Y_test_pred_std < thres]
    print(thres,":" ,error_filtered.max(), error_filtered.size, error_filtered.size/true_error.size)
    
# is able to filer size
records = []

for thres in [6, 5, 4, 3, 2, 1, 0.5, 0.2, 0.1]:
    error_filtered = true_error[Y_test_pred_std < thres]
    
    print('variance threshold: ', thres)
    print('filtered samples: ', true_error.size - error_filtered.size, 
          ';', 1 - error_filtered.size/true_error.size)
    print()
    
    record = []
    for e_thres in [30, 20, 10, 5, 2, 1, 0.5, 0.2]:
        mask = (error_filtered > e_thres)
        print(e_thres, ':',mask.sum(), ';', mask.mean())
        
        record.append(mask.sum())
        
    records.append(record)
        
    print('-'*20)
    
records = np.array(records)
records
mask = Y_test_pred_std < 0.5 # estimated error filter
#mask = true_error < 1 # true error filter

plt.plot([-10, 15], [-10, 15], c='k')
plt.scatter(Y_test[mask], Y_test_pred_mean[mask], alpha=0.2, s=4);
#plt.hexbin(Y_test, Y_test_pred_mean, bins='log')

plt.xlim(-10, 15); plt.ylim(-10, 15)
model = Sequential([
    Dense(100, input_shape=(127,)),
    Activation('relu'),
    Dropout(0.2),
    Dense(10),
    Activation('relu'),
    Dropout(0.2),
    Dense(1),
])

model.compile(optimizer='adam',
              loss='mse')
model.fit(X_train, Y_train, epochs=10, batch_size=128)
%time Y_train_pred = model.predict(X_train)
r2_score(Y_train, Y_train_pred)
%time Y_test_pred = model.predict(X_test)
r2_score(Y_test, Y_test_pred)
plt.scatter(Y_test, Y_test_pred, alpha=0.3)
import keras

def model_with_dropout():
    '''
    Dropout at both training and test time
    https://github.com/keras-team/keras/issues/9412#issuecomment-366487249
    '''
    inputs = keras.Input(shape=(127,))
    o1 = Dense(100)(inputs)
    o1d = Dropout(0.2)(o1, training=True)
    a1 = Activation('relu')(o1d)
    o2 = Dense(10)(a1)
    o2d = Dropout(0.2)(o2, training=True)
    a2 = Activation('relu')(o2d)
    outputs = Dense(1)(a2)
    model = keras.Model(inputs, outputs)
    
    return model
model_d = model_with_dropout()

model_d.compile(optimizer='adam',
              loss='mse')
model_d.fit(X_train, Y_train, epochs=10, batch_size=128)
%%time 
pred_ensemble = []
for _ in range(10):
    pred_ensemble.append(model_d.predict(X_test))
    print('.', end='')
ensemble_mean = np.mean(pred_ensemble, axis=0)
ensemble_std = np.std(pred_ensemble, axis=0)
ensemble_mean.shape
r2_score(ensemble_mean, Y_test)
true_error_d = np.abs(Y_test - ensemble_mean)
plt.scatter(ensemble_std, true_error, alpha=0.1)
plt.xlabel('ensemble variance')
plt.ylabel('true error')
stats.pearsonr(ensemble_std, true_error)
stats.spearmanr(ensemble_std, true_error)
for thres in [6, 5, 4, 3, 2, 1, 0.5, 0.2, 0.1]:
    error_filtered = true_error[ensemble_std < thres]
    
    print('variance threshold: ', thres)
    print('filtered samples: ', true_error.size - error_filtered.size, 
          ';', 1 - error_filtered.size/true_error.size)
    print()
    
    record = []
    for e_thres in [30, 20, 10, 5, 2, 1, 0.5, 0.2]:
        mask = (error_filtered > e_thres)
        print(e_thres, ':',mask.sum(), ';', mask.mean())
        
    print('-'*20)
