# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import os

print(os.listdir("../input"))

from scipy.stats import zscore

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout

import pandas as pd

import io

import requests

import numpy as np

from sklearn import metrics

from sklearn.model_selection import KFold

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

import lightgbm as lgb

from bayes_opt import BayesianOptimization

import warnings, gc

import re



# Any results you write to the current directory are saved as output.



# Any results you write to the current directory are saved as output.
def multi_merge(left,right,*args):

    start = args[0]

    for i in range(1,len(args)):

        start = start.merge(args[i], how = 'left', left_on = left, right_on = right)

    return start



def to_xy(df, target):

    result = []

    for x in df.columns:

        if x != target:

            result.append(x)

    # find out the type of the target column.  Is it really this hard? :(

    target_type = df[target].dtypes

    target_type = target_type[0] if hasattr(

        target_type, '__iter__') else target_type

    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.

    if target_type not in (np.int64, np.int32):

        # Classification

        dummies = pd.get_dummies(df[target])

        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)

    # Regression

    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)



def clean_data(df,year):

    column_list = list(df.columns)

    A_list = [column for column in column_list if re.match(r'^[A][0-9]{2,10}',column) or re.match(r'^[a][0-9]{2,10}',column)]

    N_list = [column for column in column_list if re.match(r'^[N][0-9]{2,10}',column) or re.match(r'^[n][0-9]{2,10}',column)]

    O_list = [column for column in column_list if (column not in A_list and column not in N_list and column not in ['STATE','STATEFIPS','state','statefips'])]

    if 'n1' in O_list:

        N1 = 'n1'

    else:

        N1 = 'N1'

    special = []

    pair = []

    for name in A_list:

        if 'N' + name[1:] not in N_list and 'n' + name[1:] not in N_list:

            special.append(name)

        else:

            pair.append(name)

            if 'n' + name[1:] in N_list:

                pair.append('n' + name[1:])

            else:

                pair.append('N' + name[1:])

    special = O_list + special

    pair = pair + ['zipcode',N1]

    df_special = df.loc[:,special]

    df_pair = df.loc[:,pair]

    for column in list(df_special.columns):

        if column != 'zipcode':

            df_special[column] = df_special[column]/df_special[N1]

    df_special = df_special.groupby(by='zipcode').agg('mean')

    #print(df_special)

    for name in N_list:

        if name[0] == 'N':

            A_name = 'A' + name[1:]

        else:

            A_name = 'a' + name[1:]

        df_pair[A_name] = df_pair[A_name]/df_pair[name]

        df_pair[name] = df_pair[name]/df_pair[N1]

    #print(df_pair)

    df_pair = df_pair.groupby(by='zipcode').mean().drop(N1,axis=1)

    #print(df_pair)

    df_result = df_special.merge(df_pair,how='inner',left_on='zipcode',right_on='zipcode').drop([N1,'agi_stub'],axis=1)

    #print(df_pair.shape)

    #print(df_special.shape)

    #print(df_result.shape)

    result_name_list = [name  if name == 'zipcode' else name + '_' + year for name in list(df_result.columns)]

    df_result.columns = result_name_list

    return df_result
df_2016 = pd.read_csv('../input/2016-soi-tax-stats/16zpallagi.csv')

df_2016 = df_2016.loc[(df_2016['zipcode'] != 0) & (df_2016['zipcode'] != 99999)].reset_index(drop=True)

df_2015 = pd.read_csv('../input/individual-income-tax-statistics/15zpallagi.csv')

df_2015 = df_2015.loc[(df_2015['zipcode'] != 0) & (df_2015['zipcode'] != 99999)].reset_index(drop=True)

df_2014 = pd.read_csv('../input/individual-income-tax-statistics/2014.csv')

df_2014 = df_2014.loc[(df_2014['zipcode'] != 0) & (df_2014['zipcode'] != 99999)].reset_index(drop=True)

df_2013 = pd.read_csv('../input/individual-income-tax-statistics/2013.csv')

df_2013 = df_2013.loc[(df_2013['zipcode'] != 0) & (df_2013['zipcode'] != 99999)].reset_index(drop=True)
df_2014.head()
df_2016_result = clean_data(df_2016,'2016').fillna(0)

df_2016_result.head()
df_2014_result = clean_data(df_2014,'2014').fillna(0)

df_2014_result.head()
def normalization(x):

    min_x = min(x)

    max_x = max(x)

    return np.array([(i-min_x)/(max_x-min_x) for i in x])
def zscore_transformation(df):

    for name in df.columns:

        df[name] = zscore(df[name])

    return df.dropna(axis=1).dropna(axis=0)
df_2016_result = zscore_transformation(df_2016_result)

df_2014_result = zscore_transformation(df_2014_result)
df_2014_result.shape
df_train = pd.read_csv('../input/midterm/train.csv')

df_test = pd.read_csv('../input/midterm/test.csv')

df_train = multi_merge('zipcode','zipcode',df_train,df_2014_result).drop(['id','zipcode'],axis=1)

df_test = multi_merge('zipcode','zipcode',df_test,df_2014_result).drop(['id','zipcode'],axis=1)

x,y = to_xy(df_train,'score')
df_train.dropna(axis=1)

df_train.shape
kf = KFold(5)

oos_y = []

oos_pred = []

fold = 0

for train,test in kf.split(x):

    fold += 1

    print('Fold #{}'.format(fold))

    x_train = x[train]

    y_train = y[train]

    x_test = x[test]

    y_test = y[test]

    

    model = Sequential()

    model.add(Dense(20, input_dim=x.shape[1], activation='relu'))

    model.add(Dense(30, activation='relu'))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=20, verbose=1, mode='auto')

    checkpointer = ModelCheckpoint(filepath="midterm_model1_best", verbose=0, save_best_only=True)

    model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpointer],verbose=0,epochs=250)

    

    pred = model.predict(x_test)

    oos_y.append(y_test)

    oos_pred.append(pred)

    score = np.sqrt(metrics.mean_squared_error(pred,y_test))

    print("Fold score (RMSE): {}".format(score))



oos_y = np.concatenate(oos_y)

oos_pred = np.concatenate(oos_pred)

score = metrics.mean_squared_error(oos_pred,oos_y)

print("Final score (RMSE): {}".format(np.sqrt(score)))



df_test = pd.read_csv('../input/midterm/test.csv')

df_test = multi_merge('zipcode','zipcode',df_test,df_full_detail).drop(['id','zipcode'],axis=1)

true_test = df_test.values.astype(np.float32)

pred_test = model.predict(true_test)

pred_test = model.predict(true_test)

final_test_score = np.concatenate(pred_test)



df_test['score'] = final_test_score

df_test = df_test.loc[:,['id','score']]

df_test.to_csv('csv_to_submit_regularization.csv', index = False)