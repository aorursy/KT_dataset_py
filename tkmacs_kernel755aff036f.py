# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



pd.set_option('display.max_columns', 100)

train = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv').drop(['joined'],axis = 1)

train = train.fillna('0')

'''test = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv').drop(['joined'],axis = 1)

test = test.fillna('0')'''

for i in train.columns[65:91]:

    train[i]=train[i].map(eval)

'''for j in test.columns[64:90]:

    test[j] = test[j].map(eval)'''



categorical_columns  = [c for c in train.columns if (train[c].dtype == 'object')&(c != 'value_eur')]   
train_sp = train

#test_sp = test

cat = pd.DataFrame()

#cat_test = pd.DataFrame()



for i in categorical_columns:

    cat = pd.concat([cat, train[i].str.split(', ', expand=True)], axis=1)

    #cat_test = pd.concat([cat_test,test[i].str.split(', ',expand=True)],axis=1)

    train_sp = train_sp.drop(i,axis = 1)

    #test_sp = test_sp.drop(i,axis = 1)

cat.columns = range(cat.shape[1])

#cat_test.columns = range(cat_test.shape[1])



train_sp = pd.concat([train_sp, cat],axis = 1)

train_sp = train_sp.fillna(0)

'''test_sp = pd.concat([test_sp,cat_test],axis = 1)

test_sp =test_sp.fillna(0)'''



categorical_columns_sp  = [c for c in train_sp.columns if (train_sp[c].dtype == 'object')&(c != 'value_eur')]  

dummy = pd.get_dummies(train_sp[categorical_columns_sp], drop_first=True)

train_dummy = pd.concat([train_sp.drop(categorical_columns_sp,axis = 1),dummy],axis = 1)

'''categorical_columns_sp_test  = [c for c in test_sp.columns if (test_sp[c].dtype == 'object')&(c != 'value_eur')]  

dummy = pd.get_dummies(test_sp[categorical_columns_sp_test], drop_first=True)

test_dummy = pd.concat([test_sp.drop(categorical_columns_sp_test,axis = 1),dummy],axis = 1)'''
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

def met_rmse(y_test,y_pred):

    return np.sqrt(mean_squared_error(y_test,y_pred))
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_validate



model = lgb.LGBMRegressor()

score_func = {'rmse': make_scorer(met_rmse)}



X = train.drop(categorical_columns,axis = 1).drop(['value_eur'],axis = 1)

Y = np.log(train['value_eur']+1)



scores = cross_validate(model, X, Y, scoring=score_func)

print('rmse:', scores['test_rmse'])

print('rmse:', scores['test_rmse'].mean())





model = lgb.LGBMRegressor()

score_func = {'rmse': make_scorer(met_rmse)}



X_dummy = train_dummy.drop(['value_eur'],axis = 1)



scores = cross_validate(model, X_dummy, Y, scoring=score_func)

print('rmse:', scores['test_rmse'])

print('rmse:', scores['test_rmse'].mean())
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler,StandardScaler

pca = PCA(n_components=30)

pc = train_dummy.drop(['value_eur'],axis = 1)

mms = MinMaxScaler()

pc_ms = mms.fit_transform(pc)

pc_ms = pca.fit_transform(pc_ms)



#pd.DataFrame(pca.explained_variance_ratio_).head(100)

#plt.plot(pca.explained_variance_ratio_)
model = lgb.LGBMRegressor()

score_func = {'rmse': make_scorer(met_rmse)}



Y = np.log(train['value_eur']+1)

pca = PCA(n_components=30)

pc = train_dummy.drop(['value_eur'],axis = 1)

mms = StandardScaler()

pc_ms = mms.fit_transform(pc)



pc_ms = pca.fit_transform(pc_ms)



scores = cross_validate(model, pc_ms, Y, scoring=score_func)

print('rmse:', scores['test_rmse'])

print('rmse:', scores['test_rmse'].mean())
train = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv').drop(['joined'],axis = 1)

train = train.fillna('0')

test = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv').drop(['joined'],axis = 1)

test = test.fillna('0')



all_df = pd.concat([train.drop('value_eur', axis=1), test])

for j in test.columns[64:90]:

    all_df[j] = all_df[j].map(eval)



categorical_columns_all  = [c for c in all_df.columns if (all_df[c].dtype == 'object')&(c != 'value_eur')]



all_sp = all_df

cat_all = pd.DataFrame()



for i in categorical_columns_all:

    cat_all = pd.concat([cat_all, all_df[i].str.split(', ', expand=True)], axis=1)

    all_sp = all_sp.drop(i,axis = 1)

cat_all.columns = range(cat_all.shape[1])

cat_all = cat_all.fillna(0)



all_sp = pd.concat([all_sp, cat_all],axis = 1)

all_sp = all_sp.fillna(0)



categorical_columns_sp_all  = [c for c in all_sp.columns if (all_sp[c].dtype == 'object')&(c != 'value_eur')]  

dummy = pd.get_dummies(all_sp[categorical_columns_sp_all], drop_first=True)

all_dummy = pd.concat([all_sp.drop(categorical_columns_sp_all,axis = 1),dummy],axis = 1)
X_dummy = all_dummy[:len(train)].to_numpy()

Y = train['value_eur'].to_numpy()

Y = np.log(Y+1)

t = all_dummy[len(train):].to_numpy()
model = lgb.LGBMRegressor()

model.fit(X_dummy,Y)

p = model.predict(t)
submit_df = pd.read_csv('../input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)

submit_df['value_eur'] = np.exp(p)-1

submit_df.to_csv('submission_split_dummy.csv')
pca = PCA(n_components=30)

pc = all_dummy#.drop(['value_eur'],axis = 1)

mms = StandardScaler()

pc_ms = mms.fit_transform(pc)



pc_ms = pca.fit_transform(pc_ms)





train_sp = pc_ms[:len(train)]

test_sp = pc_ms[len(train):]
model = lgb.LGBMRegressor()

model.fit(train_sp,Y)

p = model.predict(test_sp)
submit_df = pd.read_csv('../input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)

submit_df['value_eur'] = np.exp(p)-1

submit_df.to_csv('submission_split_dummy_pca_all.csv')
train = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv').drop(['joined'],axis = 1)

train = train.fillna('0')

test = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv').drop(['joined'],axis = 1)

test = test.fillna('0')



all_df = pd.concat([train.drop('value_eur', axis=1), test])

for j in test.columns[64:90]:

    all_df[j] = all_df[j].map(eval)



categorical_columns_all  = [c for c in all_df.columns if (all_df[c].dtype == 'object')&(c != 'value_eur')]



all_sp = all_df

cat_all = pd.DataFrame()



for i in categorical_columns_all:

    cat_all = pd.concat([cat_all, all_df[i].str.split(', ', expand=True)], axis=1)

    all_sp = all_sp.drop(i,axis = 1)

cat_all.columns = range(cat_all.shape[1])

cat_all = cat_all.fillna(0)





cat_all

all_sp = pd.concat([all_sp, cat_all],axis = 1)

all_sp = all_sp.fillna(0)



categorical_columns_sp_all  = [c for c in all_sp.columns if (all_sp[c].dtype == 'object')&(c != 'value_eur')]  

dummy = pd.get_dummies(all_sp[categorical_columns_sp_all], drop_first=True)



num = all_sp.drop(categorical_columns_sp_all,axis = 1)

pca = PCA(n_components=20)

#pc = num.drop(['value_eur'],axis = 1)

mms = StandardScaler()

pc_ms = mms.fit_transform(num)

pc_ms = pca.fit_transform(pc_ms)



all_dummy = np.concatenate([pc_ms, dummy.values], 1)

train_sp = all_dummy[:len(train)]

test_sp = all_dummy[len(train):]
model = lgb.LGBMRegressor()

score_func = {'rmse': make_scorer(met_rmse)}



X = train_sp

Y = np.log(train['value_eur']+1)



scores = cross_validate(model, X, Y, scoring=score_func)

print('rmse:', scores['test_rmse'])

print('rmse:', scores['test_rmse'].mean())
model = lgb.LGBMRegressor()

model.fit(train_sp,Y)

p = model.predict(test_sp)
submit_df = pd.read_csv('../input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)

submit_df['value_eur'] = np.exp(p)-1

submit_df.to_csv('submission_split_dummy_pca.csv')
train_cat
test_sp
X.shape
dummy.values.shape
pc_ms.shape
all_dummy.shape
Y.shape
train
num