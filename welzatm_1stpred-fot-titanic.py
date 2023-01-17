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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import Perceptron

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



%matplotlib inline
import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

train.head()
train.shape
#I am about starting my course on Data visualization.Below has been achieved reading through visualization personally
plt.figure(figsize=(20,5))

sns.barplot(x=train['Pclass'], y= train['Survived'], hue=train['Pclass'], capsize=0.2)

plt.title("Survived vs. Pclass")

plt.show();
plt.figure(figsize=(20,5))

sns.barplot(x=train['Sex'], y= train['Survived'], hue=train['Sex'], capsize=0.2)

plt.title("Survived vs. Sex")

plt.show();
plt.figure(figsize=(20,5))

sns.barplot(x=train['SibSp'], y= train['Survived'], hue=train['SibSp'], capsize=0.2)

plt.title("Survived vs. SibSp")

plt.show();
plt.figure(figsize=(20,5))

sns.barplot(x=train['Parch'], y= train['Survived'], hue=train['Parch'], capsize=0.2)

plt.title("Survived vs. Parch")

plt.show();
plt.figure(figsize=(20,5))

sns.barplot(x=train['Embarked'], y= train['Survived'], hue=train['Embarked'], capsize=0.2)

plt.title("Survived vs. Embarked")

plt.show();
plt.hist(x = train['Fare'] )

plt.show()
plt.hist(x = train['Age'] )

plt.show()
#Let us check for mising data

train_missing_data = train.isnull()

train_missing_data.sum()
#Fill missing data 

train['Age'].fillna(train['Age'].mean() , inplace = True)

train['Embarked'].fillna(train['Embarked'].mode() , inplace = True)
train.head()
#Drop Cabin and ticket due to lots of mising data and inconsistencies.Name probably shouldn't affect prediction

train.drop(['Name' , 'Cabin' , 'Ticket'] , axis = 1 , inplace = True)
train.head()
#Binning for Normalization

pclass_bin = np.linspace(min(train['Pclass']) , max(train['Pclass']) , 4)

group_name = [ '1stClass' , '2ndClass' , '3rdClass']

train['Pclass-binned'] = pd.cut(train['Pclass'] , pclass_bin , labels = group_name , include_lowest = True )

train['Pclass-binned'].head()
train.head()
train['Age'].describe()
age_bin = np.linspace(min(train['Age']) , max(train['Age']) , 4)

group_name = ['Youth' , 'Adult' , 'Elders']

train['Age-binned'] = pd.cut(train['Age'] , age_bin , labels = group_name , include_lowest = True )

train['Age-binned'].head()
train.head()
#Convert CAategorical to Numeric variables

pclass_dummy = pd.get_dummies(train['Pclass-binned'])

pclass_dummy
age_dummy = pd.get_dummies(train['Age-binned'])

age_dummy
sex_dummy = pd.get_dummies(train['Sex'])

sex_dummy
train.drop(['Age' , 'Sex' , 'Pclass'] , axis = 1 , inplace = True)
train.head()


train = pd.concat([train , pclass_dummy , age_dummy , sex_dummy] , axis = 1 )
train.head()
emb_dummy = pd.get_dummies(train['Embarked'])

emb_dummy
train = pd.concat([train , emb_dummy] , axis = 1)
train.head()
train.drop(['Embarked' , 'Age-binned' , 'Pclass-binned'] , axis = 1 , inplace = True)
train_missing_data = train.isnull()

train_missing_data.sum()
train.head()
features = ['SibSp' , 'Parch' , 'Fare' , '1stClass' , '2ndClass' , '3rdClass' , 'Youth' , 'Adult' , 'Elders' , 'female' , 'male' , 'C' , 'Q' , 'S']
Z_train = train[features]

Y_train = train['Survived']
#Modelling 

gnb = GaussianNB()

gnb_cross_val = cross_val_score(gnb, Z_train ,Y_train , cv=4)

gnb_cross_val 
gnb_cross_val = gnb_cross_val.mean()

gnb_cross_val
gnb.fit(Z_train , Y_train)

gnb_log = round(gnb.score(Z_train , Y_train)*100 , 2)

gnb_log
gb = GradientBoostingClassifier()

gb.fit(Z_train ,Y_train)

gb_cross_val = cross_val_score(gb, Z_train ,Y_train , cv=4)

gb_cross_val
gb_cross_val = gb_cross_val.mean()

gb_cross_val
gb_log = round(gb.score(Z_train , Y_train) * 100 , 2)

gb_log
rfc = RandomForestClassifier()

rfc.fit(Z_train ,Y_train)

rfc_cross_val = cross_val_score(rfc , Z_train , Y_train , cv = 4)

rfc_cross_val
rfc_cross_val = rfc_cross_val.mean()

rfc_cross_val
rfc_log = round(rfc.score(Z_train , Y_train) * 100 , 2)

rfc_log
dct = DecisionTreeClassifier()

dct.fit(Z_train ,Y_train)

dct_cross_val = cross_val_score(dct , Z_train , Y_train , cv = 4)

dct_cross_val
dct_cross_val = dct_cross_val.mean()

dct_cross_val
dct_log = round(dct.score(Z_train , Y_train) * 100 , 2)

dct_log
mlp = MLPClassifier()

mlp.fit(Z_train ,Y_train)

mlp_cross_val = cross_val_score(mlp , Z_train , Y_train , cv = 4)

mlp_cross_val
mlp_cross_val = mlp_cross_val.mean()

mlp_cross_val
mlp_log = round(mlp.score(Z_train , Y_train) * 100 , 2)

mlp_log
sgd = SGDClassifier()

sgd.fit(Z_train ,Y_train)

sgd_cross_val = cross_val_score(sgd , Z_train , Y_train)

sgd_cross_val 
sgd_cross_val = sgd_cross_val.mean()

sgd_cross_val
sgd_log = round(sgd.score(Z_train , Y_train)*100 , 2)

sgd_log
per = Perceptron()

per.fit(Z_train ,Y_train)

per_cross_val = cross_val_score(mlp , Z_train , Y_train , cv = 4)

per_cross_val
per_cross_val = per_cross_val.mean()

per_cross_val
per_log = round(per.score(Z_train , Y_train) * 100 , 2)

per_log
logreg = LogisticRegression()

logreg.fit(Z_train , Y_train)

logreg_cross_val = cross_val_score(logreg , Z_train , Y_train , cv = 4)

logreg_cross_val
logreg_cross_val = logreg_cross_val.mean()

logreg_cross_val
logreg_log = round(logreg.score(Z_train , Y_train) * 100 , 2)

logreg_log
sv = SVC()

sv.fit(Z_train , Y_train)

sv_cross_val = cross_val_score(sv , Z_train , Y_train , cv = 4)

sv_cross_val
sv_cross_val = sv_cross_val.mean()

sv_cross_val
sv_log = round(sv.score(Z_train , Y_train) * 100 , 2)

sv_log
test = pd.read_csv('../input/test.csv')
test.shape
test.head()
#Check for missing data

test_missing_data = test.isnull()

test_missing_data.sum()
#Fil missing data

test['Age'].fillna(test['Age'].mean(), inplace = True)

test['Fare'].fillna(test['Fare'].mean(), inplace = True)
test.drop(['Name' , 'Cabin' , 'Ticket'] , axis = 1 , inplace = True)
test.head()
age_bin = np.linspace(min(test['Age']) , max(test['Age']) , 4)

group_name = ['Youth' , 'Adult' , 'Elders']

test['Age-binned'] = pd.cut(test['Age'] , age_bin , labels = group_name , include_lowest = True)

test['Age-binned'].head()
pclass_bin = np.linspace(min(test['Pclass']) , max(test['Pclass']) , 4)

group_name = ['1stClass' , '2ndClass' , '3rdClass']

test['Pclass-binned'] = pd.cut(test['Pclass'] , pclass_bin , labels = group_name , include_lowest = True)

test['Pclass-binned'].head()
test.head()
age_dummy = pd.get_dummies(test['Age-binned'])

age_dummy.head()
pclass_dummy = pd.get_dummies(test['Pclass-binned'])

pclass_dummy.head()
emb_dummy = pd.get_dummies(test['Embarked'])

emb_dummy.head()
test.drop(['Embarked' , 'Age' , 'Pclass' , 'Age-binned' , 'Pclass-binned'] , axis = 1 , inplace = True)
test.head()
test = pd.concat([test , age_dummy , emb_dummy , pclass_dummy] , axis = 1 )
test.head()
test.shape
train.shape
sex_dummy = pd.get_dummies(test['Sex'])

sex_dummy.head()
test = pd.concat([test , sex_dummy] , axis = 1)
test.head()
test.drop('Sex' , axis = 1 , inplace = True)
test.head()
test.shape
Z_test = test[features]
models = pd.DataFrame( {'Model' : ['RandomForest','GradientBoost','DecisionTree', 'SVC','Perceptron','MLPClassifier','LogisticsReg', 'GaussianNB','SGDClassifier' ] , 'CrossValScore' : [rfc_cross_val, gb_cross_val, dct_cross_val, sv_cross_val, per_cross_val, mlp_cross_val, logreg_cross_val, gnb_cross_val, sgd_cross_val] , 'AccScore' : [rfc_log, gb_log , dct_log ,sv_log , per_log , mlp_log ,logreg_log , gnb_log , sgd_log]})
models
gb = GradientBoostingClassifier()

gb.fit(Z_train , Y_train)

Y_test = gb.predict(Z_test)

Y_test[:5]
submission = pd.DataFrame( {'PassengerId' : test['PassengerId'] , 'Survived' : Y_test})
submission.to_csv('Submission.csv' , index = False )