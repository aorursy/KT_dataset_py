# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer
from fancyimpute import KNN
Train = pd.read_csv('../input/train.csv')
Train = shuffle(Train)
Train.head()
#print(Train.columns)
y = Train['Survived']
X1 = Train[['Pclass','Age','SibSp','Parch','Fare']]
X2 = Train[['Sex','Embarked']]
X3 = pd.get_dummies(X2,columns=['Sex','Embarked'])
X3.tail()
X1.isnull().any()
X4 = pd.concat([X1,X3],axis=1)
X4.isnull().sum()

NullAge = X4['Age'].notnull()
NullAge.count()
X5 = X4 [NullAge]
yModified = y[NullAge]
yModified.describe()

X6 = X5.copy()
X6 = (X6-X5.mean())/X5.std()
X6.describe()
t = int(X6.Age.count()*0.8)
X7 = X6[:t]
y7 = yModified[:t]
X7.describe()
X_crossval = X6[t:]
y_crossval = yModified[t:]
X_crossval.describe()
logreg = LogisticRegression()
logreg.fit(X7,y7)
y_crossvalpred = logreg.predict(X_crossval)
k = y_crossvalpred ==y_crossval
print(k.mean())
Test = pd.read_csv('../input/test.csv')
Test.isnull().any()
print(Test.columns)
X_test1 = Test[['Pclass','Age','SibSp','Parch','Fare']]
X_test2 = Test[['Sex','Embarked',]]
X_test3 = pd.get_dummies(X_test2,columns=['Sex','Embarked'])
X_test3.tail()
X_test3.isnull().any()
X_test4 = pd.concat([X_test1,X_test3],axis=1)
X_test4.describe()
X_test4.isnull().sum()

X_test4_numeric = X_test4.as_matrix()
X_test5 = pd.DataFrame(KNN(3).complete(X_test4_numeric))
#X_test5 = X_test4.dropna(axis=0,how='any')
X_test5.describe()
X_test5.index = X_test4.index
X_test5.columns = X_test4.columns
X_test5.isnull().sum()

X_test5.head()
X_test6 = (X_test5-X5.mean())/X5.std()
X_test6.isnull().any()
Y_predtest = logreg.predict(X_test6)
Y_predtest1 = pd.DataFrame(Y_predtest).rename(columns={0:'Survived'})
Y_predtest1.head()
Y_predtest2 = pd.concat([Test['PassengerId'],Y_predtest1],axis=1)
Y_predtest2.head()
Y_predtest2.to_csv('out.csv',index=False)