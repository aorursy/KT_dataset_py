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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head(10)

train_data.tail(10)



train_data.query('Cabin == Cabin').shape



test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
len(train_data.loc[(train_data.Sex == 'female') & (train_data.Survived == 1)]["Survived"])

train_data.loc[:,["Name", "Age", "Pclass"]]

train_data[["Name", "Age", "Pclass"]]



women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)



men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived=", rate_men)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

actual = model.predict(X)

predictions = model.predict(X_test)



#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#output.to_csv('my_submission.csv', index=False)

print ( "X data accuracy: " , round(accuracy_score(y, actual) * 100, 2), "%" )

#print("Your submission was successfully saved!")
##############################################

######    변수 추가 Age Fare Cabin     ########

# X data accuracy:  84.4 %

#     77.5%

##############################################



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"] # class 변수

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



X['Age'] = train_data['Age']

X['Fare'] = train_data['Fare']

X['Cabin'] = np.where(train_data['Cabin'].isna()==True,0,1)

#np.where(조건, 참값, 거짓값)



X.groupby('Cabin').count()



X_test['Age'] = test_data['Age']

X_test['Fare'] = test_data['Fare']

X_test['Cabin'] = np.where(test_data['Cabin'].isna()==True,0,1)

#X.isna().sum()

X['Age'] = X['Age'].fillna(X['Age'].mean())



X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())

X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].mean())



model = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1)

model.fit(X, y)

actual = model.predict(X)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print ( "X data accuracy: " , round(accuracy_score(y, actual) * 100, 2), "%" )

print("Your submission was successfully saved!")
X
#[age if age > 10 else 10 for age in X['Age']]



#new_age = []

#for age in X['Age']:

#    if age > 10:

#        new_age.append(age)

#    else:

#        new_age.append(10)

    

# Logistic regression

# Gradient Boosting

# SVM

# Deep learning



############################

### Logistic regression 

### X data accuracy:  80.36 %

## test data : 75%

############################





from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



#model = LogisticRegression(C=1000.0, random_state=7)

#model.fit(X, y)



#actual = model.predict(X)

#predictions = model.predict(X_test)



#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#output.to_csv('my_submission.csv', index=False)



#print ( "X data accuracy: " , round(accuracy_score(y, actual) * 100, 2), "%" )

#print("Your submission was successfully saved!")
##########################

### XGBOOST  

### X data accuracy:  93.71%

## test data : 73.2%

##########################



from xgboost import plot_importance

from xgboost import XGBClassifier



#model = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5)

#model.fit(X, y)

#actual = model.predict(X)

#predictions = model.predict(X_test)



#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#output.to_csv('my_submission.csv',index=False)



#print ( "X data accuracy: " , round(accuracy_score(y, actual) * 100, 2), "%" )

#print("Your submission was successfully saved!")
############################

###          SVM         

### X data accuracy:  78.68 %

## test data : 76.5%

############################





from sklearn import svm

from sklearn.metrics import accuracy_score



#model = svm.SVC(kernel = 'linear')

#model.fit(X, y)



#actual = model.predict(X)

#predictions = model.predict(X_test)



#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#output.to_csv('my_submission.csv', index=False)



#print ( "X data accuracy: " , round(accuracy_score(y, actual) * 100, 2), "%" )

#print("Your submission was successfully saved!")
######################################

###          Deep Learning 

###      X data accuracy:  78.68 % 

## test data : 

######################################



import tensorflow as tf

import keras

from keras.models import Sequential

from keras.layers.core import Dense

np.random.seed(7)



#model = Sequential()

#model.add(Dense(255, input_shape=(8,), activation='relu'))

#model.add(Dense((1), activation='sigmoid'))

#model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

#model.summary()



from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

#hist = model.fit(X, y,epochs=100)



import matplotlib.pyplot as plt



#predictions = model.predict(X_test)



#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions.ravel()})

#output.to_csv('my_submission.csv', index=False)



#print ( "X data accuracy: " , round(accuracy_score(y, actual) * 100, 2), "%" )

#print("Your submission was successfully saved!")
