# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgbm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, BatchNormalization, Activation 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import gc
# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/kaggletutorial/covertype_train.csv')
test = pd.read_csv('../input/kaggletutorial/covertype_test.csv')
train.shape
test.shape
train_index = train.shape[0]
original_all_data = pd.concat([train, test])
all_data = original_all_data.copy()
original_all_data['Soil_Type']
for col in all_data.loc[:,all_data.dtypes=='object'].columns:
    all_data[col] = all_data[col].factorize()[0]
all_data['Soil_Type']
all_data2 = pd.concat([train, test])
for col in all_data2.loc[:,all_data2.dtypes=='object'].columns:
    le = LabelEncoder()
    all_data2[col] = le.fit_transform(all_data2[col])
all_data2['Soil_Type']
unique_soil_type = sorted(original_all_data['Soil_Type'].unique())
for index, soil in enumerate(unique_soil_type):
    print(soil, original_all_data.loc[original_all_data['Soil_Type']==soil ].shape[0], 
          all_data2.loc[all_data2['Soil_Type']==index ].shape[0]) 
unique_soil_type = sorted(original_all_data['Soil_Type'].unique())
for index, soil in enumerate(unique_soil_type):
    print(soil, original_all_data.loc[original_all_data['Soil_Type']==soil ].shape[0], 
          all_data.loc[all_data['Soil_Type']==index ].shape[0]) 
all_data2 = pd.concat([train, test])
le = LabelEncoder()
%timeit(le.fit_transform(all_data2['Soil_Type']))
%timeit(all_data2['Soil_Type'].factorize()[0])
all_data = pd.concat([train, test])
for col in all_data.loc[:, all_data.dtypes=='object'].columns:
    all_data[col] = all_data[col].factorize()[0]
train_df = all_data.iloc[:train_index]
test_df = all_data.iloc[train_index:]
y_value = train_df['Cover_Type']
del train_df['Cover_Type'], train_df['ID']

del test_df['Cover_Type'], test_df['ID']
lgbm_param =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    "learning_rate": 0.06,
    "num_leaves": 16,
    "max_depth": 6,
    "colsample_bytree": 0.7,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "nthread":8
}
NFOLD = 5
folds = StratifiedKFold(n_splits= NFOLD, shuffle=True, random_state=2018)

total_score = 0
best_iteration = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, y_value)):
    train_x, train_y = train_df.iloc[train_idx], y_value.iloc[train_idx]
    valid_x, valid_y = train_df.iloc[valid_idx], y_value.iloc[valid_idx]
    
    evals_result_dict = {} 
    dtrain = lgbm.Dataset(train_x, label=train_y)
    dvalid = lgbm.Dataset(valid_x, label=valid_y)
  
    clf = lgbm.train(lgbm_param, train_set=dtrain, num_boost_round=3000, valid_sets=[dtrain, dvalid],
                           early_stopping_rounds=200, evals_result=evals_result_dict, verbose_eval=100)
    
    predict = clf.predict(valid_x)
    cv_score = log_loss(valid_y, predict )
    total_score += cv_score
    best_iteration = max(best_iteration, clf.best_iteration)
    print('Fold {} LogLoss : {}'.format(n_fold + 1, cv_score ))
    lgbm.plot_metric(evals_result_dict)
    plt.show()
print("Best Iteration", best_iteration)
print("Total LogLoss", total_score/NFOLD)
dtrain = lgbm.Dataset(train_df, label=y_value)
clf = lgbm.train(lgbm_param, train_set=dtrain, num_boost_round=best_iteration)
predict = clf.predict(test_df)
submission = pd.read_csv('../input/kaggletutorial/sample_submission.csv')
submission["Cover_Type"] = predict
submission.to_csv('lightgbm_baseline_{:.5f}.csv'.format(total_score/NFOLD), index=False)
all_data = pd.concat([train, test])
all_data = pd.concat([train, test])
category_feature = []
for col in all_data.loc[:, all_data.dtypes=='object'].columns:
    all_data[col] = all_data[col].factorize()[0]
    category_feature.append(col)
category_feature
all_data.isnull().sum()
sns.distplot(all_data.loc[all_data['Aspect'].notnull(),'Aspect'])
plt.show()
sns.distplot(all_data['Aspect'].fillna(all_data['Aspect'].mean()))
plt.show()
all_data['Aspect'].fillna(all_data['Aspect'].mean(), inplace=True)
train_df = all_data.iloc[:train_index]
test_df = all_data.iloc[train_index:]
numerical_feature = list(set(train_df.columns) - set(category_feature) - set(['Cover_Type','ID']))
numerical_feature
sc = StandardScaler()
train_df[numerical_feature] = sc.fit_transform(train_df[numerical_feature])
test_df[numerical_feature] = sc.transform(test_df[numerical_feature] )
y_value = train_df['Cover_Type']
del train_df['Cover_Type'], train_df['ID']

del test_df['Cover_Type'], test_df['ID']
def keras_model(input_dims):
    model = Sequential()
    
    model.add(Dense(input_dims, input_dim=input_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(input_dims//2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    # output layer (y_pred)
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    # compile this model
    model.compile(loss='binary_crossentropy', # one may use 'mean_absolute_error' as alternative
                  optimizer='adam', metrics=['accuracy'])
    return model

def keras_history_plot(history):
    plt.plot(history.history['loss'], 'y', label='train loss')
    plt.plot(history.history['val_loss'], 'r', label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()
model = keras_model(train_df.shape[1])
callbacks = [
        EarlyStopping(
            patience=10,
            verbose=10)
    ]


NFOLD = 5
folds = StratifiedKFold(n_splits= NFOLD, shuffle=True, random_state=2018)

total_score = 0
best_epoch = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, y_value)):
    train_x, train_y = train_df.iloc[train_idx], y_value.iloc[train_idx]
    valid_x, valid_y = train_df.iloc[valid_idx], y_value.iloc[valid_idx]
    
    history = model.fit(train_x.values, train_y.values, nb_epoch=30, batch_size = 64, validation_data=(valid_x.values, valid_y.values), 
                        verbose=1, callbacks=callbacks)
    
    keras_history_plot(history)
    predict = model.predict(valid_x.values)
    null_count = np.sum(pd.isnull(predict) )
    if null_count > 0:
        print("Null Prediction Error: ", null_count)
        predict[pd.isnull(predict)] = predict[~pd.isnull(predict)].mean()
    
    cv_score = log_loss(valid_y, predict )
    total_score += cv_score
    best_epoch = max(best_epoch, np.max(history.epoch))
    print('Fold {} LogLoss : {}'.format(n_fold + 1, cv_score ))
print("Best Epoch: ", best_epoch)
print("Total LogLoss", total_score/NFOLD)
history = model.fit(train_df.values, y_value.values, nb_epoch=best_epoch, batch_size = 64, verbose=1)
predict = model.predict(test_df.values)
null_count = np.sum(pd.isnull(predict) )
if null_count > 0:
    print("Null Prediction Error: ", null_count)
    predict[pd.isnull(predict)] = predict[~pd.isnull(predict)].mean()
submission = pd.read_csv('../input/kaggletutorial/sample_submission.csv')
submission["Cover_Type"] = predict
submission.to_csv('neuralnetwork_baseline_{:.5f}.csv'.format(total_score/NFOLD), index=False)