import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder



import os

print(os.listdir("../input"))

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from xgboost import plot_importance





train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])

test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')
train.head()
test.head()
test.info()
train.isnull().sum()
import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from xgboost import plot_importance



pd.options.display.max_columns=100

train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])

test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')



def feature_engineering(data):

    data['Date'] = pd.to_datetime(data['Dates'].dt.date)

    data['Day'] = data['Dates'].dt.day

    data['DayOfWeek'] = data['Dates'].dt.weekday

    data['Month'] = data['Dates'].dt.month

    data['Year'] = data['Dates'].dt.year

    data['Hour'] = data['Dates'].dt.hour

    data['Minute'] = data['Dates'].dt.minute

    data['Block'] = data['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)

    data.drop(columns=['Dates','Date','Address'], inplace=True)

    return data

train = feature_engineering(train)

test = feature_engineering(test)

train.drop(columns=['Descript','Resolution'], inplace=True)
train.head()
test.head()
le1 = LabelEncoder()

train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])

test['PdDistrict'] = le1.transform(test['PdDistrict'])



le2 = LabelEncoder()

X = train.drop(columns=['Category'])

y= le2.fit_transform(train['Category'])
train.head()
test.head()
X.head()
import xgboost as xgb

train_xgb = xgb.DMatrix(X, label=y)

test_xgb  = xgb.DMatrix(test)
params = {

    'max_depth': 6,  # the maximum depth of each tree

    'eta': 0.3,  # the training step for each iteration

    'num_boost_rounds' : 150 ,

    'silent': 1,  # logging mode - quiet

    'objective': 'multi:softprob',  # error evaluation for multiclass training

    'eval_metric' : 'mlogloss',

    'learning_data' : 0.07,

    'num_class': 39,

}
CROSS_VAL = False

if CROSS_VAL:

    print('Doing Cross-validation ...')

    cv = xgb.cv(params, train_xgb, nfold=3, early_stopping_rounds=10, metrics='mlogloss', verbose_eval=True)

    cv
SUBMIT = not CROSS_VAL

if SUBMIT:

    print('Fitting Model ...')

    m = xgb.train(params, train_xgb, 10)

    res = m.predict(test_xgb)

    submission = pd.DataFrame(res, columns=le2.inverse_transform

                              (np.linspace(0, 38, 39, dtype='int16')),

                              index=test.index)

 

    submission.to_csv('submission.csv', index='Id')

    print(submission.sample(3))

else :

    print('FAIL')