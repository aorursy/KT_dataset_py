# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

train=pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test=pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

sample=pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")
#Checking for  unique values binary columns in each columns

print("Binary Unique Values . . . . .")

def bin_val(dataframe):

    l=[]

    for i in (dataframe.iloc[:,1:6]).columns:

        

        l.append([i,pd.Series(dataframe[i].unique())])

    return pd.Series(l)

print(bin_val(train))

print('\n')



#Cheking for nominal  values

print("Nominal Unique Values . . . .")



def nom_val(dataframe):

    l=[]

    for i in (dataframe.iloc[:,6:16]).columns:

        l.append([i,pd.Series(dataframe[i].unique()).count()])

    return (pd.Series(l))



print(nom_val(train))

print('\n')



#Checking for ordinal values



print("Ordinal Unique Values . . . .")

def ord_val(dataframe):

    l=[]

    for i in (dataframe.iloc[:,16:22]).columns:

        l.append([i,pd.Series(dataframe[i].unique()).count()])

    return pd.Series(l)

print(ord_val(train))

#Checking for type for columns

print('To Know Data Type Of Columns . . . . ')

typ=[]

for i in (train.iloc[:,1:24]).columns:

    typ.append(type(train[i][0]))

typ=pd.DataFrame(typ)

typ[0].value_counts()
# As we know binary values don't have relationships each other we us dummies from pandas 

train=pd.get_dummies(data=train,columns=['bin_0','bin_1','bin_2','bin_3','bin_4'])

test=pd.get_dummies(data=test,columns=['bin_0','bin_1','bin_2','bin_3','bin_4'])

  
nominal=train.iloc[:,1:6]

fig = plt.figure(figsize=(25,6))

fig.subplots_adjust(hspace=0.4,wspace=0.4)

for i in range(pd.Series(nominal.columns).count()):

    ax = fig.add_subplot(1,6,i+1)

    sns.countplot(y=nominal.iloc[:,i])
fig=plt.figure(figsize=(10,7))

fig.subplots_adjust(hspace=1,wspace=1)

nominal_2=train.iloc[:,6:11]

for i in range(pd.Series((nominal_2).columns).count()):

    ax=fig.add_subplot(3,2,i+1)

    z=pd.DataFrame(nominal_2.iloc[:,i].value_counts()).head(5)

    z.reset_index(inplace=True)

    sns.barplot(y=z['index'],x=z.iloc[:,1])
#For Nominal we are going to apply for label encoder



from sklearn.preprocessing import LabelEncoder

encoder =LabelEncoder()

def nom_val_lab_encoder(dataframe):

    for i in (dataframe.iloc[:,1:11]).columns:

        e=encoder.fit(dataframe[i])

        dataframe[i]=pd.Series(e.transform(dataframe[i]))

    return dataframe
train=nom_val_lab_encoder(train)

test=nom_val_lab_encoder(test)
#train.iloc[:,11:17]

train=pd.get_dummies(data=train,columns=['ord_0','ord_1','ord_2','ord_3','ord_4','ord_5'])

test=pd.get_dummies(data=test,columns=['ord_0','ord_1','ord_2','ord_3','ord_4','ord_5'])
y_train = train['target'].copy()

x_train = train.drop('target', axis=1)

del train



x_test = test.copy()

del test
from sklearn.model_selection import StratifiedKFold

import lightgbm as  lgb

from sklearn.metrics import roc_auc_score

from bayes_opt import BayesianOptimization
def train_mod(n_leaves, min_data_in_leaf, max_depth, bagging, feature, l1, l2):

    

    params = {

        'objective': 'binary',

        'metric': 'auc',

        'boosting_type': 'gbdt',

        'is_unbalance': False,

        'boost_from_average': True,

        'num_threads': 4,

        

        'n_leaves': int(n_leaves),

        'min_data_in_leaf': int(min_data_in_leaf),

        'max_depth': int(max_depth),

        'bagging' : bagging,

        'feature' : feature,

        'l1': l1,

        'l2': l2

    }

    

    scores = []

    

    cv = StratifiedKFold(n_splits=10)

    for train_idx, valid_idx in cv.split(x_train, y_train):

        

        x_train_train = x_train.iloc[train_idx]

        y_train_train = y_train.iloc[train_idx]

        x_train_valid = x_train.iloc[valid_idx]

        y_train_valid = y_train.iloc[valid_idx]

        

        lgb_train = lgb.Dataset(data=x_train_train.astype('float32'), label=y_train_train.astype('float32'))

        lgb_valid = lgb.Dataset(data=x_train_valid.astype('float32'), label=y_train_valid.astype('float32'))

        lgb_model = lgb.train(params, lgb_train, valid_sets=lgb_valid, verbose_eval=100)

        y = lgb_model.predict(x_train_valid.astype('float32'), num_iteration=lgb_model.best_iteration)

        score = roc_auc_score(y_train_valid.astype('float32'), y)

        print('Fold score:', score)

        scores.append(score)

    average_score = sum(scores) / len(scores)

    print('Average score:', average_score)

    return average_score





bounds = {

    'n_leaves': (31, 100),

    'min_data_in_leaf': (20, 100),

    'max_depth':(-1, 100),

    'bagging' : (0.1, 0.9),

    'feature' : (0.1, 0.9),

    'l1': (0, 2),

    'l2': (0, 2)

}



bo = BayesianOptimization(train_mod, bounds, random_state=42)

bo.maximize(init_points=20, n_iter=20, acq='ucb', xi=0.0, alpha=1e-6)


params = {

    'objective': 'binary',

    'metric': 'auc',

    'boosting_type': 'gbdt',

    'is_unbalance': False,

    'boost_from_average': True,

    'num_threads': 4,

    

    'bagging_fraction': 0.12033530139527615,

    'feature_fraction': 0.18631314159464357,

    'lambda_l1': 0.0628583713734685,

    'lambda_l2': 1.2728208225275608,

    'max_depth': int(30.749954088708993),

    'min_data_in_leaf': int(60.685655293176225),

    'num_leaves': int(93.62208670090041),

    

    'num_iterations': 10000,

    'learning_rate': 0.006,

    'early_stopping_round': 100

}

n_splits = 10



y = np.zeros(x_test.shape[0])



best_score = 0

best_y = []



feature_importances = []



cv = StratifiedKFold(n_splits=n_splits)

for train_idx, valid_idx in cv.split(x_train, y_train):

    

    x_train_train = x_train.iloc[train_idx]

    y_train_train = y_train.iloc[train_idx]

    x_train_valid = x_train.iloc[valid_idx]

    y_train_valid = y_train.iloc[valid_idx]

    

    lgb_train = lgb.Dataset(data=x_train_train.astype('float32'), label=y_train_train.astype('float32'))

    lgb_valid = lgb.Dataset(data=x_train_valid.astype('float32'), label=y_train_valid.astype('float32'))

    

    lgb_model = lgb.train(params, lgb_train, valid_sets=lgb_valid, verbose_eval=100)

    

    y_part = lgb_model.predict(x_test.astype('float32'), num_iteration=lgb_model.best_iteration)

    y += y_part / n_splits

    

    temp = lgb_model.predict(x_train_valid.astype('float32'), num_iteration=lgb_model.best_iteration)

    score = roc_auc_score(y_train_valid.astype('float32'), temp)

    print('Fold score:', score)

    if score > best_score:

        best_score = score

        best_y = y_part

        print('Best Y updated. Score =', score)

    

    feature_importances.append(lgb_model.feature_importance())
sample['target']=y

sample.to_csv('submission.csv',index=False)