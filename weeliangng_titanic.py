import pandas as pd

from pandas import Series, DataFrame

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.info()
test_df.info()
sns.factorplot('Sex','Survived',data=train_df,hue='Pclass')
trainMean = train_df.groupby(['Sex','Pclass']).mean()
trainMean
meanFemaleAge1 = trainMean['Age'].loc[('female',1)]

meanFemaleAge2 = trainMean['Age'].loc[('female',2)]

meanFemaleAge3 = trainMean['Age'].loc[('female',3)]



meanMaleAge1 = trainMean['Age'].loc[('male',1)]

meanMaleAge2 = trainMean['Age'].loc[('male',2)]

meanMaleAge3 = trainMean['Age'].loc[('male',3)]
train_df.loc[(train_df['Pclass']==1) & (train_df['Sex']=='male') &train_df['Age'].isnull(),'Age'] = meanMaleAge1

train_df.loc[(train_df['Pclass']==2) & (train_df['Sex']=='male') &train_df['Age'].isnull(),'Age'] = meanMaleAge2

train_df.loc[(train_df['Pclass']==3) & (train_df['Sex']=='male') &train_df['Age'].isnull(),'Age'] = meanMaleAge3



train_df.loc[(train_df['Pclass']==1) & (train_df['Sex']=='female') &train_df['Age'].isnull(),'Age'] = meanFemaleAge1

train_df.loc[(train_df['Pclass']==2) & (train_df['Sex']=='female') &train_df['Age'].isnull(),'Age'] = meanFemaleAge2

train_df.loc[(train_df['Pclass']==3) & (train_df['Sex']=='female') &train_df['Age'].isnull(),'Age'] = meanFemaleAge3
test_df.loc[(test_df['Pclass']==1) & (test_df['Sex']=='male') &test_df['Age'].isnull(),'Age'] = meanMaleAge1

test_df.loc[(test_df['Pclass']==2) & (test_df['Sex']=='male') &test_df['Age'].isnull(),'Age'] = meanMaleAge2

test_df.loc[(test_df['Pclass']==3) & (test_df['Sex']=='male') &test_df['Age'].isnull(),'Age'] = meanMaleAge3



test_df.loc[(test_df['Pclass']==1) & (test_df['Sex']=='female') &test_df['Age'].isnull(),'Age'] = meanFemaleAge1

test_df.loc[(test_df['Pclass']==2) & (test_df['Sex']=='female') &test_df['Age'].isnull(),'Age'] = meanFemaleAge2

test_df.loc[(test_df['Pclass']==3) & (test_df['Sex']=='female') &test_df['Age'].isnull(),'Age'] = meanFemaleAge3
train_df.info()
test_df.info()
def isChild(passenger):

    age,sex=passenger

    if age<17:

        return 'child'

    else:

        return sex
train_df['person']=train_df[['Age','Sex']].apply(isChild,axis=1)
test_df['person']=test_df[['Age','Sex']].apply(isChild,axis=1)
train_df['companion']=(train_df['Parch']+train_df['SibSp'])>0

test_df['companion']=(test_df['Parch']+test_df['SibSp'])>0
train_df.head()
sns.factorplot('person','Survived',data=train_df,hue='Pclass')
train_df.groupby(['Pclass','person','companion']).aggregate([np.mean,np.std])
from sklearn import tree

from sklearn.cross_validation import train_test_split

from sklearn import preprocessing
le= preprocessing.LabelEncoder()

le.fit(train_df['person'])

train_df['person_int']=le.transform(train_df['person'])

test_df['person_int']=le.transform(test_df['person'])
features=['Age','person_int','Pclass','companion']
clf = tree.DecisionTreeClassifier()
clf.fit(train_df[features],train_df[['Survived']])
predictions=clf.predict(test_df[features])
test_df['Survived']=predictions
test_df[['PassengerId','Survived']].to_csv('submission.csv',index=False) #70%
from sklearn.neural_network import MLPClassifier

import sklearn
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(7,7,7), random_state=1)
clf.fit(train_df[features],train_df[['Survived']].values.ravel())
predictions=clf.predict(test_df[features])
test_df['Survived']=predictions
test_df[['PassengerId','Survived']].to_csv('submission.csv',index=False)