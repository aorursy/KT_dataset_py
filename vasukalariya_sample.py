# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df['Age'].head()
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
sex = pd.get_dummies(df['Sex'],drop_first=True)
embark = pd.get_dummies(df['Embarked'],drop_first=True)
df.head()
X = df.drop('Survived',axis = 1)
X.head()
df.drop(['Sex','Embarked','Name','Ticket','Cabin','PassengerId'],axis=1,inplace=True)
df = pd.concat([df,sex,embark],axis=1)
df.head()
from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators = 100,learning_rate=0.002,gamma = 5)
X = df.drop('Survived',axis = 1)
X.head()
clf.fit(X,df['Survived'])
testdf = pd.read_csv('../input/test.csv')
testdf.isna().sum()
testdf[testdf['Fare'].isnull()][['Pclass','Age','PassengerId']]
t = testdf[testdf['Pclass'] == 3]
t[(t['Age'] > 40) & (t['Age'] < 70)]['Fare'].mean()
testdf['Age'] = testdf[['Age','Pclass']].apply(impute_age,axis=1)
testdf.set_value(152,col='Fare',value=15)
testdf.isna().sum()
tsex = pd.get_dummies(testdf['Sex'],drop_first=True)
tembark = pd.get_dummies(testdf['Embarked'],drop_first=True)
pid = testdf['PassengerId']
testdf.drop(['Sex','Embarked','Name','Ticket','Cabin','PassengerId'],axis=1,inplace=True)
testdf = pd.concat([testdf,tsex,tembark],axis=1)
testdf.head()
testdf.isna().sum()
pred = clf.predict(testdf)
pred.shape
clf2 = XGBClassifier(n_estimators = 100,learning_rate=0.002,gamma = 7)
clf2.fit(X,df['Survived'])
pred2 = clf2.predict(testdf)
pred2
test = pd.read_csv('../input/test.csv',index_col=False)
test = test.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
#test.set_index('PassengerId',inplace = True)
test
pr = pd.Series(pred)
test['Survived'] = pr
test.set_index('PassengerId',inplace=True)
test
test.to_csv('submission')

