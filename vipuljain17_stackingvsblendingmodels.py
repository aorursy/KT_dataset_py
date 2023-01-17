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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import StackingClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.metrics import f1_score,mean_squared_error
data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

data.shape
data.nunique().sort_values(ascending=False).head(5)
data.drop(["Name","Ticket","Cabin"],axis=1,inplace=True)

data.head(),data.shape
data.info()
data.isnull().sum().sort_values(ascending = False).head()
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')



#Fill  those NAN values with mean of the feature column

data.Age = imp.fit_transform(data[['Age']]).ravel()



imp2 = SimpleImputer(missing_values=np.nan,strategy='most_frequent')

data.Embarked = imp2.fit_transform(data[['Embarked']]).ravel()



data.Age.mean(),data.Age.std(),data.Age.isnull().sum()
data.isnull().sum().sort_values(ascending = False).head(2)
plt.hist(data.Age,bins=10)

plt.title("Age distribution")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.show()
table = pd.crosstab(data['Survived'],data['Sex']);table
data.groupby('Sex').Survived.mean() #Females were the greater amount of surviours
#Which Class had greater survival rate

table = pd.crosstab(data['Survived'],data['Pclass'])

table
##SEX and Pclass

sns.catplot(x='Pclass',hue='Sex',col='Survived',data = data,kind = 'count',height=5,aspect=.8)

plt.show()
data.head()
data['relatives'] = data['SibSp'] + data['Parch']

data.loc[data['relatives'] > 0, 'not_alone'] = 0

data.loc[data['relatives'] == 0, 'not_alone'] = 1

data['not_alone'] = data['not_alone'].astype(int)
data.tail(2)
data = data.join(pd.get_dummies(data['Sex']))

data = data.join(pd.get_dummies(data['Embarked']))

data.tail()
test_data.drop(["Name","Ticket","Cabin"],axis=1,inplace=True)



test_data['relatives'] = test_data['SibSp'] + test_data['Parch']

test_data.loc[test_data['relatives'] > 0, 'not_alone'] = 0

test_data.loc[test_data['relatives'] == 0, 'not_alone'] = 1

test_data['not_alone'] = test_data['not_alone'].astype(int)



test_data.Age = imp.transform(test_data[['Age']]).ravel()

test_data.Embarked = imp2.transform(test_data[['Embarked']]).ravel()



test_data = test_data.join(pd.get_dummies(test_data['Sex']))

test_data = test_data.join(pd.get_dummies(test_data['Embarked']))



test_data.shape
"""Don't select dummy varibles to get trap"""



f = ['Pclass','Age','SibSp','Parch','Fare','relatives','not_alone','female','C','Q']



len(f)
test_data.Fare.fillna(method = 'ffill',inplace=True)
X = data[f]

X_test = test_data[f]



y = data['Survived']
X.shape,X_test.shape
level0 = list()

level0.append(('lr', make_pipeline(StandardScaler(),LogisticRegression())))

level0.append(('cart', DecisionTreeClassifier()))

level0.append(('svm', make_pipeline(StandardScaler(),SVC())))

level0.append(('random_forest', RandomForestClassifier()))
level1 = LogisticRegression()
model = StackingClassifier(estimators=level0, final_estimator=level1)
model.fit(X,y)
pred_y = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred_y})

output.to_csv('my_submission.csv', index=False)

print("Submitted successfully!")