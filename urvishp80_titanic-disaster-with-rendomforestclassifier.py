# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df_train=pd.read_csv('../input/train.csv')

#print(df.head())



#loading test data

df_test=pd.read_csv('../input/test.csv')

print(df_test.head())

##finding nan values in age

#print(pd.isnull(df_train['Age']))

df_train.dropna(how='any',axis=1,subset=df_train['Survived'])
y=df_train['Survived']

#replacing nan values with mean of the age

mean_age=np.mean(df_train['Age'])

#print(mean_age)

age=df_train['Age'].fillna(mean_age)

real_age=np.array(age)

print(real_age.size)



#replacing nan values in age for test data

mean_age=np.mean(df_test['Age'])

#print(mean_age)

age=df_test['Age'].fillna(mean_age)

test_age=np.array(age)

#encoding the sex with Lableencoder for training data

from sklearn.preprocessing import LabelEncoder

sex=df_train['Sex']

enc=LabelEncoder()

label_encoder=enc.fit(sex)

print ("Categorical classes:", label_encoder.classes_)

integer_classes = label_encoder.transform(label_encoder.classes_)

print ("Integer classes:", integer_classes)

sex_real = label_encoder.transform(sex)

print((sex_real.size))



#encoding sex for testing data

sex1=df_test['Sex']

enc=LabelEncoder()

label_encoder=enc.fit(sex)

print ("Categorical classes:", label_encoder.classes_)

integer_classes = label_encoder.transform(label_encoder.classes_)

print ("Integer classes:", integer_classes)

test_sex = label_encoder.transform(sex1)

print(type(test_sex))



#encoding passanger clss for trainig data

pclass=df_train['Pclass']

#print(pclass.isnull())

from sklearn.preprocessing import LabelEncoder

pclass=LabelEncoder().fit_transform(pclass)

print(pclass.size)



#encoding passanger class for test data

test_pclas=df_test['Pclass']

#print(pclass.isnull())

from sklearn.preprocessing import LabelEncoder

test_pclass_real=LabelEncoder().fit_transform(test_pclas)

#print(test_pclass)
#frames=[real_age,sex_real,pclass]

X_train=np.stack([real_age,sex_real,pclass],axis=1)

print(X_train.shape)
from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier



clf1=LogisticRegression()

clf2=svm.SVC(kernel='rbf',C=1,gamma=1)

clf3=RandomForestClassifier()

clfs=[clf1,clf2,clf3]

for clf in clfs:

    clf.fit(X_train,y)

    accuracy=clf.score(X_train,y)

    print(accuracy)

from sklearn.model_selection import GridSearchCV

param_dict={'n_estimators':[10,20,40],'max_depth':[10,11,12,13,14,15]}

grid_clf1=GridSearchCV(clf3,param_grid=param_dict,scoring='accuracy',n_jobs=-1)

grid_clf1.fit(X_train,y)

print(grid_clf1.best_estimator_.score(X_train,y))

print(grid_clf1.best_estimator_)
X_test=np.stack([test_age,test_sex,test_pclass_real],axis=1)

print(X_test.shape)
#y_pred=grid_clf1.best_estimator_.predict(X_test)

#print(y_pred)

y_pred=clf2.predict(X_test)

passenger_id=df_test['PassengerId']

#res=pd.Series(y_pred,index=df_test['PassengerId'])

#print(res)

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': y_pred } )

test.shape

test.head()

test.to_csv( 'titanic_pred.csv' , index = False )