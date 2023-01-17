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

import numpy as np

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier



from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import seaborn as sns
X=pd.read_csv("/kaggle/input/titanic/train.csv")

data_test=pd.read_csv("/kaggle/input/titanic/test.csv")

XX=X.copy()
X.head()
X.info()
sns.heatmap(X.isna())
X.isna().sum()
sns.distplot(X.Age)
sns.distplot(X.Fare)
X.Age.mean()

m=X.Age.mean()
X.Age.std()

var=X.Age.std()
#ll=np.random.normal(1,1,1000)

#X["Age"]=X["Age"].fillna(np.random.choice(ll))

#X["Age"]=X["Age"].fillna(X["Age"].mean())
#X["Age"].fillna(lambda x: random.choice(X[X["Age"] != np.nan]["Age"]), inplace =True)

X["Age"]=X["Age"].mask(X["Age"].isnull(), np.random.normal(m, var, size=X["Age"].shape))
for i in X["Age"]:

    print(i)
1-X.Survived.sum()/X.Survived.size
col=["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

data=X[col]

data.Age=data.Age.fillna(data.Age.mean())

data.dropna()

data["family"]=data["SibSp"]+data["Parch"]+1

data=data.drop(columns=["SibSp","Parch"],axis=1)

data=pd.get_dummies(data,columns=["Sex","Embarked"])



X,y=data.drop("Survived",axis=1),data["Survived"]

estimator=RandomForestClassifier(n_estimators=500)

model=make_pipeline(RobustScaler(),estimator)

model.fit(X,y)

model.score(X,y)
X_test=data_test[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]

X_test["family"]=X_test["SibSp"]+X_test["Parch"]+1

X_test=X_test.drop(columns=["SibSp","Parch"],axis=1)

X_test.Age=X_test.Age.fillna(data.Age.mean())

X_test.Fare=X_test.Fare.fillna(data.Fare.mean())

X_test=pd.get_dummies(X_test,columns=["Sex","Embarked"])

Y_test=model.predict(X_test)
data=pd.DataFrame({"PassengerId":data_test.PassengerId,"Survived":Y_test})

data.to_csv("my_submission_x.csv",index=False)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.15)
model.score(X_test,y_test)

