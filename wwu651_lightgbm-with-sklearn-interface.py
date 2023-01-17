import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")
train=pd.read_csv('../input/act_train.csv',parse_dates=['date'])

test=pd.read_csv('../input/act_test.csv',parse_dates=['date'])

people=pd.read_csv('../input/people.csv',parse_dates=['date'])
train.head(1)
test.head(1)
people.head(1)
train=pd.merge(train,people,on='people_id')

test=pd.merge(test,people,on='people_id')
train.info()
test.info()
train.outcome.plot(kind='hist')

plt.show()
def checkMissing(df):

    column_names=[]

    null_count=[]

    for i in df.columns:

        if df[i].isnull().sum()!=0:

            column_names.append(i)

            null_count.append(df[i].isnull().sum())

    if len(null_count)==0:

        print('There is no missing values!')

    else:

        plt.figure(figsize=(8,4))

        sns.barplot(x=null_count,y=column_names,color='C0')

        plt.show()

checkMissing(train)
train.char_38.nunique()
plt.hist(train[train['outcome']==1]['char_38'],color='C1',alpha=0.7)

plt.hist(train[train['outcome']==0]['char_38'],color='C0',alpha=0.7)

plt.show()

print('Number of 0 value:',train[train .char_38==0]['char_38'].count())
train.describe(include=['object']).transpose()
len(set(test['group_1'])-set(train['group_1']))
for d in ['date_x', 'date_y']:

    print('Start of ' + d + ': ' + str(train[d].min().date()))

    print('  End of ' + d + ': ' + str(train[d].max().date()))

    print('Range of ' + d + ': ' + str(train[d].max() - train[d].min()) + '\n')
# Impute the missing values with type 0

for i in train.columns:

    if np.dtype(train[i])==np.dtype('object'):

        train[i].fillna('type 0',inplace=True)
for i in test.columns:

    if np.dtype(test[i])==np.dtype('object'):

        test[i].fillna('type 0',inplace=True)
checkMissing(train)

checkMissing(test)
# Mean and Median outcome group by group_1 for train

outcomeMeanGroupbyGroup_1=train.groupby(['group_1'])['outcome'].mean().to_frame().reset_index()

outcomeMedianGroupbyGroup_1=train.groupby(['group_1'])['outcome'].median().to_frame().reset_index()

dict_outcomeMeanGroupbyGroup_1=dict(zip(outcomeMeanGroupbyGroup_1['group_1'],outcomeMeanGroupbyGroup_1['outcome']))

dict_outcomeMedianGroupbyGroup_1=dict(zip(outcomeMedianGroupbyGroup_1['group_1'],outcomeMedianGroupbyGroup_1['outcome']))

train['outcomeMeanGroupbyGroup_1']=train['group_1'].map(lambda x: dict_outcomeMeanGroupbyGroup_1.get(x))

train['outcomeMedianGroupbyGroup_1']=train['group_1'].map(lambda x: dict_outcomeMedianGroupbyGroup_1.get(x))



# Mean and Median outcome group by group_1 for test

test['outcomeMeanGroupbyGroup_1']=test['group_1'].map(lambda x: dict_outcomeMeanGroupbyGroup_1.get(x))

test['outcomeMedianGroupbyGroup_1']=test['group_1'].map(lambda x: dict_outcomeMedianGroupbyGroup_1.get(x))
checkMissing(test)
# Impute missing value for test

test['outcomeMeanGroupbyGroup_1']=test.groupby(['activity_category'])['outcomeMeanGroupbyGroup_1'].transform(lambda x: x.fillna(x.mean()))

test['outcomeMedianGroupbyGroup_1']=test.groupby(['activity_category'])['outcomeMedianGroupbyGroup_1'].transform(lambda x: x.fillna(x.mean()))
checkMissing(test)
def featureEngineering(df):

    # feature engineering for dates

    listDate=['year', 'month', 'week', 'day', 'dayofweek', 'dayofyear','is_month_end', 'is_month_start', 'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start']

    for n in listDate:

        df[n.upper()]=df['date_x'].map(lambda x: getattr(x,n))

        df[n.upper()]=df['date_y'].map(lambda x: getattr(x,n))

    

    # Extract numbers from cateforical data and convert boolean to int

    for i in df.columns:

        if np.dtype(df[i])==np.dtype('object') and i not in ['activity_id','people_id']:

            df[i]=df[i].map(lambda x:int(x.split(' ')[1]))

        elif np.dtype(df[i])==np.dtype('bool'):

            df[i]=df[i].map(lambda x:int(x))



    return df
# Feature Engineering for train and test

train=featureEngineering(train)

test=featureEngineering(test)
# Check feature number of train and test after feature engineering

missingfeatures=list(set(train.columns.tolist())-set(test.columns.tolist()))

missingfeatures.remove('outcome')

print(missingfeatures)
train.head()
X_train=train.drop(['outcome','date_x','date_y','activity_id','people_id'],axis=1).copy()

y_train=train['outcome'].copy()

X_test=test.drop(['date_x','date_y','activity_id','people_id'],axis=1).copy()
X_train.head()
# To save some time. I used reduced dataset.
# X_train_demo=X_train.iloc[:50000,:].copy()

# y_train_demo=y_train.iloc[:50000].copy()
from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

from sklearn.metrics import roc_auc_score
# Cross validate model with Kfold cross validation

kfold=StratifiedKFold(n_splits=2)
# I've narrow down the range of the parameters after some tryings in order to save some time.

lgbm = LGBMClassifier(random_state=8)



lgbm_param_grid = {'num_leaves' : [2,3],

                'learning_rate' :   [0.004,0.005,0.006,0.007],

                'n_estimators': [50,100]}



lgbm = GridSearchCV(lgbm,param_grid = lgbm_param_grid, cv=kfold, scoring="accuracy",n_jobs=2, verbose = 1)



lgbm.fit(X_train,y_train)

# How to use the output of GridSearch? Please see https://datascience.stackexchange.com/questions/21877/how-to-use-the-output-of-gridsearch

# Best score

print(lgbm.best_score_)

print(lgbm.best_params_)
results = pd.DataFrame({ 'activity_id' : test['activity_id'].values, 

                       'outcome': lgbm.predict_proba(X_test)[:,1] })
results.to_csv('red_hat_LightGBM.csv',index=False)