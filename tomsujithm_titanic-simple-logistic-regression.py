# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
traindf = pd.read_csv("../input/titanic/train.csv")
testdf = pd.read_csv("../input/titanic/test.csv")
df = pd.concat([traindf, testdf], sort = False)

df.info()
df.head()
df.shape
df.describe()
import seaborn as sns
sns.heatmap(df.corr(),annot=True)
sns.pairplot(df)
df.isnull().sum().sort_values(ascending=False)
sns.heatmap(df.isnull(),yticklabels=False)
sns.set_style('whitegrid')
sns.countplot(x='Survived',data = df,hue='Sex')
sns.countplot(x='Survived',data = df,hue = 'Pclass')
sns.distplot(df['Age'].dropna(),bins=10)
sns.countplot(x='SibSp',data = df)
df.loc[(df['Age'].isnull()) & (df['Pclass']==1),'Age']=int(df[['Age']][df['Pclass']==1].mean())

df.loc[(df['Age'].isnull()) & (df['Pclass']==2),'Age']=int(df[['Age']][df['Pclass']==2].mean())

df.loc[(df['Age'].isnull()) & (df['Pclass']==3),'Age']=int(df[['Age']][df['Pclass']==3].mean())
df.loc[(df['Fare'].isnull()) & (df['Pclass']==1),'Fare']=int(df[['Fare']][df['Pclass']==1].mean())

df.loc[(df['Fare'].isnull()) & (df['Pclass']==2),'Fare']=int(df[['Fare']][df['Pclass']==2].mean())

df.loc[(df['Fare'].isnull()) & (df['Pclass']==3),'Fare']=int(df[['Fare']][df['Pclass']==3].mean())
dummies = pd.get_dummies(df[['Sex','Embarked']],drop_first = True)
titanic = pd.concat([df,dummies],axis=1)
titanic.drop(['Sex','Embarked'],inplace = True,axis =1)
titanic.drop(['Cabin'],inplace = True,axis =1)
sns.heatmap(titanic.isnull(),yticklabels=False)
titanic.drop(['Name','Ticket'],inplace = True,axis =1)
titanic_train = titanic[titanic['Survived'].notna()]

titanic_train.info()
titanic_test = titanic[titanic['Survived'].isna()]

titanic_test.info()
y= titanic_train.pop('Survived')
X=titanic_train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=.3,random_state = 0)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
titanic_test = titanic_test.drop(['Survived'], axis = 1)
t_pred = log_reg.predict(titanic_test).astype(int)
t_pred
PassengerId = titanic_test['PassengerId']
logSub = pd.DataFrame({ 'PassengerId':PassengerId,'Survived':t_pred })

logSub.head()
logSub.to_csv("1_Logistics_Regression_Submission.csv", index = False)