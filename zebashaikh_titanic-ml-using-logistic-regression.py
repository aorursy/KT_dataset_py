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
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')

test_df=pd.read_csv('/kaggle/input/titanic/test.csv')

print(train_df.shape,test_df.shape)
train_df.head()

test_df.head()
test_df['Survived']=-999
test_df.shape
type(train_df)
dataset=pd.concat([train_df,test_df],sort=True)

dataset.head()
dataset.index=dataset['PassengerId']

dataset.drop(['PassengerId'],axis=1,inplace=True)

dataset.head()
#check missing value

dataset.isna().sum()
# Fare

dataset[dataset.Fare.isnull()]
dataset[(dataset.Pclass==3)].groupby(['Sex']).median()
dataset.Fare.fillna(7.8958,inplace=True)

dataset.isnull().sum()
#Embarked

dataset[dataset.Embarked.isnull()]
dataset.Embarked.value_counts() # highest frequency is S
dataset.groupby(['Pclass','Embarked','Sex','Survived']).Fare.median()
dataset[(dataset.Sex=='female') & (dataset.Pclass==1)].groupby(['Embarked']).median()['Fare']
dataset.Embarked.fillna('S',inplace=True)
dataset.isna().sum()
#Age

dataset.Age.isna().value_counts()
dataset.groupby(['Sex']).median()['Age']
age_sex_median = dataset.groupby('Sex').Age.transform('median')
dataset["Age"] = np.where(dataset.Age.notnull(), dataset.Age, age_sex_median )
dataset.isnull().sum()
dataset.info()
dataset.drop(['Name','Cabin','Ticket'],inplace=True,axis=1)
dataset.head()
dataset=pd.get_dummies(dataset,columns=['Sex','Embarked'],drop_first=True)
dataset.head()
from sklearn.linear_model import LogisticRegression
clean_train =dataset[dataset.Survived != -999]

clean_test=dataset[dataset.Survived ==-999]
xtrain=clean_train.drop(['Survived'],axis=1)

ytrain=clean_train['Survived']

xtest=clean_test.drop(['Survived'],axis=1)
Logreg= LogisticRegression()

Logreg.fit(xtrain,ytrain)

pred=Logreg.predict(xtest)
df=pd.DataFrame()
df['PassengerId']=xtest.index

df['Survived']=pred

df.head()

df[['PassengerId','Survived']]
df[['PassengerId','Survived']].to_csv('submission.csv',index=False)