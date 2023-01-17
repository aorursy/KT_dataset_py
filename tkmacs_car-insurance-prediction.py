# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv')

train.replace(['?'], np.nan, inplace=True)

#train['num-of-doors'] = train['num-of-doors'].map({'four':4, 'two':2})

#train['num-of-doors'] = train['num-of-doors'].astype(np.float64)

train['normalized-losses'] = train['normalized-losses'].astype(np.float64)

train['bore'] = train['bore'].astype(np.float64)

train['stroke'] = train['stroke'].astype(np.float64)

train['horsepower'] = train['horsepower'].astype(np.float64)

train['peak-rpm'] = train['peak-rpm'].astype(np.float64)

train['price'] = train['price'].astype(np.float64)

#train.dtypes

train['normalized-losses-nan'] = (train['normalized-losses'].isnull())

train['num-of-doors-nan'] = (train['num-of-doors'].isnull())

train['bore-nan'] = (train['bore'].isnull())

train['horsepower-nan'] = (train['horsepower'].isnull())

train['peak-rpm-nan'] = (train['peak-rpm'].isnull())

train['price-nan'] = (train['price'].isnull())

train['stroke-nan'] = (train['stroke'].isnull())



train['normalized-losses'] = train['normalized-losses'].fillna(train['normalized-losses'].mode()[0])

train['num-of-doors'] = train['num-of-doors'].fillna(train['num-of-doors'].mode()[0])

train['bore'] = train['bore'].fillna(train['bore'].mode()[0])

train['horsepower'] = train['horsepower'].fillna(train['horsepower'].mode()[0])

train['peak-rpm'] = train['peak-rpm'].fillna(train['peak-rpm'].mode()[0])

train['price'] = train['price'].fillna(train['price'].mode()[0])

train['stroke'] = train['stroke'].fillna(train['stroke'].mode()[0])



test = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv')

test.replace(['?'], np.nan, inplace=True)

#test['num-of-doors'] = test['num-of-doors'].map({'four':4, 'two':2})

#test['num-of-doors'] = test['num-of-doors'].astype(np.float64)

test['normalized-losses'] = test['normalized-losses'].astype(np.float64)

test['bore'] = test['bore'].astype(np.float64)

test['stroke'] = test['stroke'].astype(np.float64)

test['horsepower'] = test['horsepower'].astype(np.float64)

test['peak-rpm'] = test['peak-rpm'].astype(np.float64)

test['price'] = test['price'].astype(np.float64)



test['normalized-losses-nan'] = (test['normalized-losses'].isnull())

test['num-of-doors-nan'] = (test['num-of-doors'].isnull())

test['bore-nan'] = (test['bore'].isnull())

test['horsepower-nan'] = (test['horsepower'].isnull())

test['peak-rpm-nan'] = (test['peak-rpm'].isnull())

test['price-nan'] = (test['price'].isnull())

test['stroke-nan'] = (test['stroke'].isnull())



test['normalized-losses'] = test['normalized-losses'].fillna(test['normalized-losses'].mode()[0])

test['num-of-doors'] = test['num-of-doors'].fillna(test['num-of-doors'].mode()[0])

test['bore'] = test['bore'].fillna(test['bore'].mode()[0])

test['horsepower'] = test['horsepower'].fillna(test['horsepower'].mode()[0])

test['peak-rpm'] = test['peak-rpm'].fillna(test['peak-rpm'].mode()[0])

test['price'] = test['price'].fillna(test['price'].mode()[0])

test['stroke'] = test['stroke'].fillna(test['stroke'].mode()[0])

#train = train.fillna(-9999)

#test = test.fillna(-9999)
#pd.set_option('display.max_rows', 500)

#train.sort_values('body-style')

#train.sort_values('engine-location')

#train.sort_values('num-of-cylinders')
#train.select_dtypes(object)

#train['symboling'].unique()


def make_hist(i):

    fig = plt.figure(figsize=(25,10)) 

    p1 = fig.add_subplot(2,2,1)

    p1.hist(train[i][train.symboling == -2], bins=10, alpha = .4)

    p1.hist(train[i][train.symboling == -1], bins=10, alpha = .4)

    p1.hist(train[i][train.symboling == 0], bins=10, alpha = .4)

    p1.hist(train[i][train.symboling == 1], bins=10, alpha = .4)

    p1.hist(train[i][train.symboling == 2], bins=10, alpha = .4)

    p1.hist(train[i][train.symboling == 3], bins=10, alpha = .4)

    labels = [-2,-1,0,1,2,3]

    plt.legend(labels)
train.columns
make_hist('price')
import category_encoders as ce

import xgboost as xgb

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE,RandomOverSampler

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import SelectFromModel

def target(df,df_t,col):

    for i in col:

        #label_mean = df.groupby(i).symboling.mean()

        #df = df.assign(=df[i].map(label_mean).copy())

        label_mean = df.groupby(i).symboling.mean()

        df[i] = df[i].map(label_mean)

        df_t[i] = df_t[i].map(label_mean)

    return df,df_t
categorical_columns  = [c for c in train.columns if (train[c].dtype == 'object')&(c != 'symboling')]

train,test = target(train,test,categorical_columns)
X = train.drop('symboling',axis = 1).values

Y = train.symboling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size =0.1, random_state=1)

smote = SMOTE(k_neighbors=2)

X_s,Y_s = smote.fit_sample(X_train,Y_train)

#ros = RandomOverSampler(random_state=0)

#X_o,Y_o = ros.fit_resample(X_train,Y_train)
model = xgb.XGBRegressor()

his = model.fit(X_train,Y_train)

p = model.predict(X_test)

np.sqrt(mean_squared_error(Y_test,p))
model = xgb.XGBRegressor()

his = model.fit(X_s,Y_s)

p = model.predict(X_test)

np.sqrt(mean_squared_error(Y_test,p))
selector = SelectFromModel(estimator=xgb.XGBRegressor(),threshold='median').fit(X_s,Y_s)

X_sl = selector.transform(X_s)

X_test_sl = selector.transform(X_test)

his = model.fit(X_sl,Y_s)

p = model.predict(X_test_sl)

np.sqrt(mean_squared_error(Y_test,p))
smote = SMOTE(k_neighbors = 2)

X_u,Y_u = smote.fit_sample(X,Y)

model = xgb.XGBRegressor()

x_U = SelectFromModel(estimator=xgb.XGBRegressor(),threshold='median').fit_transform(X_u,Y_u)

his = model.fit(X_u,Y_u)
p = model.predict(test.values)
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p
submit_df.to_csv('submission_target_smote_xgb.csv')
import lightgbm as lgb
model = lgb.LGBMRegressor()

his = model.fit(X_u,Y_u)

p = model.predict(test.values)

submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p

submit_df.to_csv('submission_target_smote_lgb.csv')
train = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv')

train.replace(['?'], np.nan, inplace=True)

#train['num-of-doors'] = train['num-of-doors'].map({'four':4, 'two':2})

#train['num-of-doors'] = train['num-of-doors'].astype(np.float64)

train['normalized-losses'] = train['normalized-losses'].astype(np.float64)

train['bore'] = train['bore'].astype(np.float64)

train['stroke'] = train['stroke'].astype(np.float64)

train['horsepower'] = train['horsepower'].astype(np.float64)

train['peak-rpm'] = train['peak-rpm'].astype(np.float64)

train['price'] = train['price'].astype(np.float64)



test = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv')

test.replace(['?'], np.nan, inplace=True)

#test['num-of-doors'] = test['num-of-doors'].map({'four':4, 'two':2})

#test['num-of-doors'] = test['num-of-doors'].astype(np.float64)

test['normalized-losses'] = test['normalized-losses'].astype(np.float64)

test['bore'] = test['bore'].astype(np.float64)

test['stroke'] = test['stroke'].astype(np.float64)

test['horsepower'] = test['horsepower'].astype(np.float64)

test['peak-rpm'] = test['peak-rpm'].astype(np.float64)

test['price'] = test['price'].astype(np.float64)



train = train.fillna(-9999)

test = test.fillna(-9999)
X = train.drop('symboling',axis = 1).values

Y = train.symboling
import catboost

from sklearn.model_selection import train_test_split

model = catboost.CatBoostRegressor(iterations=50000,

                                  use_best_model=True,

                                  eval_metric = 'RMSE')

cat = [2,3,4,5,6,7,8,14,15,17]

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size =0.1, random_state=0)

'''

model.fit(X_train,y_train,

          cat_features=cat,

          eval_set=(X_test,y_test),

          plot=True

         )

'''
'''

model = catboost.CatBoostRegressor(iterations=50000)

model.fit(X,Y,cat_features=cat,plot=True)

p = model.predict(test.values)

submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p

submit_df.to_csv('submission__cat.csv')

'''
train
[train['normalized-losses'].isnull()]
train['normalized-losses'].fillna(train['normalized-losses'].mode()[0])
train['horsepower'].mode()