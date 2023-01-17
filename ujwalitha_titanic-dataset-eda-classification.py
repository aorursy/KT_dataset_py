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
from sklearn.model_selection import train_test_split,KFold,cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

from sklearn.metrics import accuracy_score
import seaborn as sns

import matplotlib.pyplot as plt

os.getcwd()
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
df = train_df.drop(['PassengerId','Name','Ticket'],axis=1)
train_df.shape
df.describe()
df.isna().sum()
from sklearn.preprocessing import normalize



fare = np.array(df['Fare']).reshape(1,-1)

fare_nor = normalize(fare)

fare_nor = fare_nor.reshape(-1,1)

fare = pd.DataFrame(fare_nor)

df['Fare'] = fare
df[df['Embarked'].isnull()]

df['Embarked'] = df['Embarked'].fillna('S')
df['Cabin'] = df['Cabin'].str[0]

df['Cabin'] = df['Cabin'].fillna('Others')

df['Cabin'] = df['Cabin'].apply(lambda x: 0 if x=='Others' else 1)
q1,q2,q3 = df['Age'].quantile([0.25,0.5,0.75])

iqr = q3-q1

upper = q3 + 1.5*(iqr)

lower = q1 - 1.5*(iqr)

upper,lower

df = df[~(df['Age']>upper)]
def age_range(x):

    if x<1:

        return '<1'

    elif x>=1 and x<=10:

        return '1-10'

    elif x>10 and x<=20:

        return '11-20'

    elif x>20 and x<=30:

        return '20-30'

    elif x>30 and x<=40:

        return '30-40'

    elif x>40 and x<=50:

        return '40-50'

    elif x>50:

        return '50+'

    else:

        return 'Empty'

        







df['Age'] = df['Age'].apply(lambda x: age_range(x))
sns.countplot(df['Pclass'])
sns.countplot(df['Embarked'])
sns.countplot(df['Cabin'])
sns.countplot(df['Age'])
df.corr()
plt.figure(figsize=(15,5))

sns.heatmap(df.corr(),annot=True)
obj_df = df.select_dtypes(include=['object']).copy()

from sklearn.preprocessing import LabelEncoder

obj_df = obj_df.apply(LabelEncoder().fit_transform)

df = df.drop(['Sex','Age','Embarked'], axis=1)

df = pd.concat([df,obj_df],axis=1)

df.head()
trainx = df.drop(['Survived'],axis=1)

trainy = df['Survived']
rfe = RandomForestClassifier()

rfe.fit(trainx,trainy)

rfe.feature_importances_

trainx = trainx.drop(['Cabin','Embarked'],axis=1)
trainx['family'] = trainx['SibSp']+trainx['Parch']+1

trainx.head()
trainx = trainx.drop(['SibSp','Parch'],axis=1)
test_df.head()
testx = test_df.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],axis=1)

testx.shape
def age_range(x):

    if x<1:

        return '<1'

    elif x>=1 and x<=10:

        return '1-10'

    elif x>10 and x<=20:

        return '11-20'

    elif x>20 and x<=30:

        return '20-30'

    elif x>30 and x<=40:

        return '30-40'

    elif x>40 and x<=50:

        return '40-50'

    elif x>50:

        return '50+'

    else:

        return 'Empty'

        







testx['Age'] = testx['Age'].apply(lambda x: age_range(x))
obj_df = testx[['Sex','Age']]

obj_df = obj_df.apply(LabelEncoder().fit_transform)

testx = testx.drop(['Sex','Age'],axis=1)

testx = pd.concat([testx,obj_df],axis=1)
testx['Fare'] = testx['Fare'].fillna(testx['Fare'].median())
from sklearn.preprocessing import normalize



fare = np.array(testx['Fare']).reshape(1,-1)

fare_nor = normalize(fare)

fare_nor = fare_nor.reshape(-1,1)

fare = pd.DataFrame(fare_nor)

testx['Fare'] = fare
testx['family'] = testx['SibSp']+testx['Parch']+1

testx = testx.drop(['SibSp','Parch'],axis=1)
testx.head()
# prepare models

models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier(max_depth=7)))

models.append(('RF', RandomForestClassifier(max_depth=7)))

# evaluate each model in turn

results = []

names = []
y_pred = []

from sklearn.model_selection import KFold,cross_val_score

import warnings

warnings.filterwarnings('ignore')

kfold = KFold(n_splits=15,random_state=1)

for name, model in models:

    bag=BaggingClassifier(base_estimator=model,random_state=1)

    bag.fit(trainx,trainy)

    results = cross_val_score(bag, trainx, trainy, cv=kfold)

    y_pred.append(bag.predict(testx))

    print(name, bag.score(trainx,trainy))

    print(name,"Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

    

    
y_pred = pd.Series(y_pred)

final_knn = pd.concat([test_df['PassengerId'],pd.Series(y_pred[0])],axis=1)

final_decision_tree = pd.concat([test_df['PassengerId'],pd.Series(y_pred[1])],axis=1)

final_rf = pd.concat([test_df['PassengerId'],pd.Series(y_pred[2])],axis=1)

final_knn = final_knn.rename(columns={0:'Survived'})

final_decision_tree = final_decision_tree.rename(columns={0:'Survived'})

final_rf = final_rf.rename(columns={0:'Survived'})
final_decision_tree.to_csv('submission_dt.csv',index=False)

final_rf.to_csv('submission_rf.csv',index=False)