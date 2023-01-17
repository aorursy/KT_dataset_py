# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib
%matplotlib inline 
import xgboost as xgb
import re
import warnings
warnings.filterwarnings('ignore')

def port(cabin) :
    if pd.isnull(cabin) :
        return 0.5
    if cabin[0] == 'E' :
        return 1.0
    try :
        cabin_number = int(re.sub(" .*","",cabin[1:]))
    except :
        return 0.5
    return 1.0 if cabin_number % 2 == 1 else 0.0

print ("D37 should be 1.0 on port",port("D37"))
print ("D37 D39 should be 1.0 on port",port("D37 D39"))
print ("A30 should be 0.0 on port",port("A30"))
print ("E1 and E2 should be 1.0 on port",port("E1"),port("E2"))
def forward(cabin) :
    if pd.isnull(cabin) :
        return 0.5
    try :
        cabin_number = int(re.sub(" .*","",cabin[1:]))
    except :
        return 0.5
    if cabin[0] == 'E' :
        return 1.0 if cabin_number > 57 else 0.0
    return 1.0 if (cabin_number > 76 and cabin_number < 100) or cabin_number > 124 else 0.0   

print ("D37 should be 0.0 on port",forward("D37"))
print ("D137 D139 should be 1.0 on port",forward("D137 D139"))
print ("A30 should be 0.0 on port",forward("A30"))
print ("E1 and E2 should be 0.0 on port",forward("E1"),forward("E2"))
print ("E60 and E70 should be 1.0 on port",forward("E60"),forward("E70"))
print ("empty should be 0.5 on port",forward(""))
def prepare_df(df):

    df['Port'] = df['Cabin'].apply(port)
    df['Forward'] = df['Cabin'].apply(forward)
    df['Has cabin'] = df['Cabin'].apply(lambda x: 0. if pd.isnull(x) else 1. )
    df['Female'] = df['Sex'].apply(lambda x: 1. if x == 'female' else 0.)

    df['P1'] = df['Pclass'].apply(lambda x: 1. if x == 1 else 0.)
    df['P2'] = df['Pclass'].apply(lambda x: 1. if x == 2 else 0.)
    df['P3'] = df['Pclass'].apply(lambda x: 1. if x == 3 else 0.)
    
    df['Miss'] = df['Name'].apply(lambda x: 1. if 'Miss.' in x else 0.)
    df['Mrs'] = df['Name'].apply(lambda x: 1. if 'Mrs.' in x else 0.)
    df['Mr'] = df['Name'].apply(lambda x: 1. if 'Mr.' in x else 0.)
    df['Master'] = df['Name'].apply(lambda x: 1. if 'Master' in x else 0.)
    df['Jr'] = df['Name'].apply(lambda x: 1. if 'Jr' in x else 0.)

    min_age = df['Age'].min()
    max_age = df['Age'].max()
    df['Child'] = df['Age'].apply(lambda x: 1.0 if x <= 13 else 0.0)
    df['Old'] = df['Age'].apply(lambda x: 1.0 if x >= 60 else 0.0)
    
    # drop unneeded columns
    df = df.drop(columns=['Embarked','Pclass','Fare','Name','Sex','Ticket','Cabin','SibSp','Parch'])
    
    for x in ['Has cabin', 'Female', 'P1', 'P2', 'P3', 'Miss', 'Mrs', 'Mr', 'Master', 'Jr', 'Child', 'Old'] :
        df[x] = df[x].astype(int)
    
    return df

df = prepare_df(pd.read_csv("../input/train.csv"))
df.head(10)
np.random.seed(1)
msk = np.random.rand(len(df)) < 0.8
print ("Mask length",msk.shape[0])
train_df = df[msk]
dev_df = df[~msk]
feature_columns_to_use = ['Female', 'Port', 'Child', 'Forward', 'Old', 'Has cabin', 'P3']

def eval_model(gbm,X,y):
    a = np.array(gbm.predict(X))
    b = np.array(y.values)
    return (a == b).mean()

def create_and_dev_model(train_df,dev_df, features=feature_columns_to_use, random_state=1 ):
    train_X = train_df[ features ]
    train_y = train_df['Survived']
    gbm = xgb.XGBClassifier(
        max_depth=3, n_estimators=5000, random_state=random_state, eval_metric='rmse'
        ).fit(train_X, train_y)
    
    train_correct = eval_model(gbm,train_X,train_y)
    
    dev_X = dev_df[ features ]
    dev_y = dev_df['Survived']
    dev_correct= eval_model(gbm,dev_X,dev_y)
    print ("Train / Dev correct %0.3f / %0.3f" % ( train_correct , dev_correct ))
    return gbm

print ("Just Female ",),
gbm = create_and_dev_model(train_df,dev_df,['Female'])
print ("Default cols",),
gbm = create_and_dev_model(train_df,dev_df)

xgb.plot_importance(gbm)
gbm = create_and_dev_model(df,df,feature_columns_to_use)
test_df = prepare_df( pd.read_csv("../input/test.csv") )
test_df['Survived'] = gbm.predict(test_df[feature_columns_to_use])
test_df.head(10)
# Let's look at where it's predicting that just being Female isn't the indicator of survival

test_df[ test_df['Female'] != test_df['Survived'] ].head(10)
