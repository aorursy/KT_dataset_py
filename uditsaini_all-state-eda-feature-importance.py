# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

plt.figure(figsize=(13,9))

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn

from subprocess import check_output



import seaborn as sns

%matplotlib inline

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
catvar=[col for col in train.columns if 'cat' in col]

contvar=[col for col in train.columns if 'cont' in col]
print('continuous variables ',catvar)

print('categorical variables ',contvar)
correlationMatrix =train[contvar+['loss']].corr().abs()

plt.subplots(figsize=(13, 9))

sns.heatmap(correlationMatrix,annot=True)

plt.show()
#loss distribution 

plt.subplots(figsize=(13, 9))

sns.kdeplot(train.loss)
from scipy import stats

sns.kdeplot(np.log1p(train.loss))
#distribution of categorical variables 

for col in catvar:

    sns.countplot(train[col])

    sns.plt.title(col)

    plt.show()
#distribution of continuse variables

for col in contvar:

    sns.distplot(train[col])

    plt.show()
#distribution of continuse variables after log transformation

for col in contvar:

    sns.distplot(np.log1p(train[col]))

    plt.show()
#check skewness in each continus variable

train[contvar].apply(lambda x: stats.skew(x))
#unique value count in each catgorial variable

for col in catvar:

    print(train[col].value_counts())
#lets find out some importent variable 

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
for col in catvar:

    train[col]=le.fit_transform(train[col])
import xgboost as xgb

#make log box cox transformation of continus variables

for col in contvar:

    train[col]=np.log(train[col])
params = {}

params['booster'] = 'gbtree'

params['objective'] = "reg:linear"

params['eval_metric'] = 'mae'

params['eta'] = 0.1

params['gamma'] = 0.5290

params['min_child_weight'] = 4.2922

params['colsample_bytree'] = 0.3085

params['subsample'] = 0.9930

params['max_depth'] = 7

params['max_delta_step'] = 0

params['silent'] = 1

params['random_state'] = 1001

dtrain = xgb.DMatrix(train[catvar+contvar], label =np.log1p(train['loss']))



evallist  = [(dtrain,'train')]

bst =xgb.train(params,dtrain,num_boost_round=250,evals=evallist,early_stopping_rounds=8, verbose_eval=10)
outfile = open("fe.map", 'w')

i = 0

for feat in catvar+contvar:

    outfile.write('{0}\t{1}\tq\n'.format(i, feat))

    i = i + 1

outfile.close()
import operator

importance = bst.get_fscore(fmap="fe.map")

importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])

df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()

df.head(25).plot()

df.head(25).plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))

plt.title('XGBoost Feature Importance')

plt.xlabel('relative importance')
#all importent features are categorical feature lets predict using this features

test=pd.read_csv("../input/test.csv")

train=pd.read_csv("../input/train.csv")
features=list(df['feature'].head(20).values)+contvar
dtrain=pd.get_dummies(train[features])

dtest=pd.get_dummies(test[features])

params = {}

params['booster'] = 'gbtree'

params['objective'] = "reg:linear"

params['eval_metric'] = 'mae'

params['eta'] = 0.06

params['gamma'] = 0.5290

params['min_child_weight'] = 4.2922

params['colsample_bytree'] = 0.5

params['subsample'] = 0.8

params['max_depth'] = 7

params['max_delta_step'] = 0

params['silent'] = 1

params['random_state'] = 1001

dtrain = xgb.DMatrix(dtrain, label =np.log1p(train['loss']))

dtest = xgb.DMatrix(dtest)

evallist  = [(dtrain,'train')]

model =xgb.train(params,dtrain,num_boost_round=500,evals=evallist,early_stopping_rounds=8, verbose_eval=10)
preds=np.expm1(model.predict(dtest))

submission = pd.DataFrame({"id":test.id,"loss":preds})

submission.to_csv('submission.csv', index=None)