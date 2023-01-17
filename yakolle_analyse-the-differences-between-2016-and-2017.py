# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMRegressor



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import warnings

warnings.filterwarnings("ignore")
# unique the train_data

def unique_train_data(rec):

    return rec.sort_values(by='transactiondate').iloc[-1]
# load features

data_dir='../input/'

prop_2016=pd.read_csv(data_dir+'properties_2016.csv',index_col='parcelid').fillna(-1)

prop_2017=pd.read_csv(data_dir+'properties_2017.csv',index_col='parcelid').fillna(-1)

prop_2017=prop_2017.loc[prop_2016.index]

print(prop_2016.shape,prop_2017.shape,np.sum(prop_2016.index!=prop_2017.index))
# compare the features between 2016 and 2017

for col in prop_2017.columns:

    s=prop_2017[col]

    s1=prop_2016[col]

    print(col,np.sum(s!=s1))
# simple preprocess

prop_2016 = prop_2016.drop(['regionidcounty', 'rawcensustractandblock','assessmentyear','propertyzoningdesc', 

                'propertycountylandusecode'],axis=1)

obj_columns = prop_2016.dtypes.loc[np.object == prop_2016.dtypes].index

for col in obj_columns:

    prop_2016[col] = prop_2016[col].apply(lambda ele: str(ele))

    prop_2016[col] = LabelEncoder().fit_transform(prop_2016[col])

    

prop_2017 = prop_2017.drop(['regionidcounty', 'rawcensustractandblock','assessmentyear','propertyzoningdesc', 

                'propertycountylandusecode'],axis=1)

obj_columns = prop_2017.dtypes.loc[np.object == prop_2017.dtypes].index

for col in obj_columns:

    prop_2017[col] = prop_2017[col].apply(lambda ele: str(ele))

    prop_2017[col] = LabelEncoder().fit_transform(prop_2017[col])

    

print('preprocess done')
# load train data

train_2016 = pd.read_csv(data_dir+'train_2016_v2.csv',index_col='parcelid')

train_2016 = train_2016.groupby('parcelid').apply(unique_train_data)

train_2016['sale_month'] = pd.to_datetime(train_2016['transactiondate']).dt.month



train_2017 = pd.read_csv(data_dir+'train_2017.csv',index_col='parcelid')

train_2017 = train_2017.groupby('parcelid').apply(unique_train_data)

train_2017['sale_month'] = pd.to_datetime(train_2017['transactiondate']).dt.month



co_index = np.intersect1d(train_2016.index,train_2017.index)

print(len(co_index))
# use the common data between 2016 and 2017 to predict the 2016 logerror

y = train_2016.loc[co_index,'logerror']

x = prop_2016.loc[co_index]

x['sale_month'] = train_2016.loc[co_index,'sale_month']



# use LGBMRegressor to fit the data and print features' importances

model = LGBMRegressor(objective='regression', n_estimators=200, learning_rate=.0125, num_leaves=24, max_depth=11,

                      max_bin=80, min_child_samples=1, min_child_weight=0, min_split_gain=4e-05, subsample=.3,

                      colsample_bytree=.45, subsample_freq=1, reg_alpha=4, reg_lambda=4, seed=0, nthread=2)

model.fit(x,y)

ims = []

for i in range(len(model.feature_importances_)):

    ims.append((x.columns[i], model.feature_importances_[i]))

ims = sorted(ims, key=lambda pair : pair[1], reverse=True)

print('which factors impact the 2016 logerror:')

for col, im in ims:

    print(col,im)
# use the common data between 2016 and 2017 to predict the 2017 logerror

y = train_2017.loc[co_index,'logerror']

x = prop_2017.loc[co_index]

x['sale_month'] = train_2017.loc[co_index,'sale_month']



# use LGBMRegressor to fit the data and print features' importances

model = LGBMRegressor(objective='regression', n_estimators=200, learning_rate=.0125, num_leaves=24, max_depth=11,

                      max_bin=80, min_child_samples=1, min_child_weight=0, min_split_gain=4e-05, subsample=.3,

                      colsample_bytree=.45, subsample_freq=1, reg_alpha=4, reg_lambda=4, seed=0, nthread=2)

model.fit(x,y)

ims = []

for i in range(len(model.feature_importances_)):

    ims.append((x.columns[i], model.feature_importances_[i]))

ims = sorted(ims, key=lambda pair : pair[1], reverse=True)

print('which factors impact the 2017 logerror:')

for col, im in ims:

    print(col,im)
# use the gap of features to predict the gap of logerror

y = train_2017.loc[co_index,'logerror'] - train_2016.loc[co_index,'logerror']

x = prop_2017.loc[co_index] - prop_2016.loc[co_index]

x['sale_month'] = train_2017.loc[co_index,'sale_month'] - train_2016.loc[co_index,'sale_month']



# use LGBMRegressor to fit the data and print features' importances

model = LGBMRegressor(objective='regression', n_estimators=200, learning_rate=.0125, num_leaves=24, max_depth=11,

                      max_bin=80, min_child_samples=1, min_child_weight=0, min_split_gain=4e-05, subsample=.3,

                      colsample_bytree=.45, subsample_freq=1, reg_alpha=4, reg_lambda=4, seed=0, nthread=2)

model.fit(x,y)

ims = []

for i in range(len(model.feature_importances_)):

    ims.append((x.columns[i], model.feature_importances_[i]))

ims = sorted(ims, key=lambda pair : pair[1], reverse=True)

print('which factors cause the change of logerror:')

for col, im in ims:

    print(col,im)