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
data = pd.read_csv('/kaggle/input/titanic/train.csv')

gender=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
data.head()
data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
data.head()
import seaborn as sb
sb.heatmap(data.isna())
sb.boxplot(x='Pclass',y='Age',data=data)
meanage = data.groupby('Pclass').mean()['Age']

def impute_age(cols):

    age = cols[0]

    pclass = cols[1]

    if not pd.isnull(age):

        return age

    else:

        return meanage[pclass]
data['Age'] = data[['Age','Pclass']].apply(impute_age,axis=1)
data.head()
data = pd.get_dummies(columns=['Sex','Embarked'],data=data,drop_first=True)
data.drop('PassengerId',axis=1,inplace=True)
data.head()
from sklearn.model_selection import train_test_split

X = data.drop(['Survived','Fare'],axis=1)

y = data['Survived']

X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train,y_train)
y_predict = lg.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
test = pd.get_dummies(columns=['Sex','Embarked'],data=test,drop_first=True)
test.head()
sb.heatmap(test.isna())
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
test.drop('PassengerId',axis=1,inplace=True)
test.drop('Fare',axis=1,inplace=True)
test.head()
X_train.head()
y_predict = lg.predict(test)
y_predict
predicted = pd.read_csv('/kaggle/input/titanic/test.csv')
predicted.head()
predicted['Survived']=y_predict
predicted.head()
sb.countplot(data=predicted,x='Survived',hue='Sex')
output = predicted[['PassengerId','Survived']]
output.head()
gender.head()
output.to_csv('submission.csv')