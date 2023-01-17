# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# We read the file 

data=pd.read_csv('/kaggle/input/titanic/train.csv')
!pip install feature-engine
data.head()
data.isnull().sum()
data.info()
data_copied=data
# We use crosstab on Pclass and Survived column to find the number of passangers that survived, and which class they belong to.

pd.crosstab(data['Pclass'],data['Survived'])
# From the plot we can clearly understand a lot of the 3rd class passengers didnt survive

sns.countplot(x='Pclass',hue='Survived',data=data)
# To check there chance of survival if they had any family member.

temp1=pd.crosstab([data['SibSp'],data['Parch']],data['Survived'])
temp1.plot(kind='bar',figsize=(12,9))
# We combine the sibiling and parents column to find out if there is higher chance of survival, when there are more members.

data['Total Family Members']=data['SibSp']+data['Parch']+1
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
data['Total Family Members']=data['Total Family Members'].map(family_map)
plt.figure(figsize=(12,9))

sns.countplot(hue='Survived',x='Total Family Members',data=data)

plt.legend(loc='upper right')
temp2=pd.crosstab(data['Sex'],data['Survived'])
# To see if females have a higher chance of survival.

temp2.plot(kind='bar')
# Plot to check if the boarding location results in the survival of a person

sns.countplot(x='Embarked',hue='Survived',data=data)
# We create a new feature deck

data['Deck']=data['Cabin'].dropna().str[0]



data['Deck'].value_counts(dropna=False)
# We fill the NA Values of the Deck column with M so as to indicate missing values



data['Deck']=data['Deck'].fillna('M')
X=data
X=X.drop(columns=['PassengerId','SibSp','Parch','Cabin','Ticket','Survived'])
y=data['Survived']
# We split the dataset into train and test data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# We fill in the missing NAN values for embarked column

X_train['Embarked'].fillna(X['Embarked'].mode()[0],inplace=True)
# We fill NAN values of the age column with random values

random_sample_train=X_train['Age'].dropna().sample(X_train['Age'].isnull().sum(),random_state=0)

random_sample_test=X_train['Age'].dropna().sample(X_test['Age'].isnull().sum(),random_state=0)



random_sample_train.index = X_train[X_train['Age'].isnull()].index

random_sample_test.index=X_test[X_test['Age'].isnull()].index



X_train.loc[X_train['Age'].isnull(), 'Age'] = random_sample_train

X_test.loc[X_test['Age'].isnull(), 'Age'] = random_sample_test
# we bin the age column by finding the min and max value in the train set

inter_value=int(X_train['Age'].max()-X_train['Age'].min())



min_value=int(X_train['Age'].min())



max_value=int(X_train['Age'].max())



range_value=int(inter_value/10)



interval=[x for x in range(min_value,max_value+range_value,range_value)]



X_train['Age_disc']=pd.cut(x=X_train['Age'],bins=interval,include_lowest=True)



X_test['Age_disc']=pd.cut(x=X_test['Age'],bins=interval,include_lowest=True)
# We discretise the fare column in equal widths



from feature_engine.discretisers import EqualWidthDiscretiser



ewd = EqualWidthDiscretiser(bins=10,variables=['Fare'])



ewd.fit(X_train)



X_train=ewd.transform(X_train)



X_test=ewd.transform(X_test)
# coverting the age_disc to string to perform ordinal encoding

X_train['Age_disc']=X_train['Age_disc'].astype('str')

X_test['Age_disc']=X_test['Age_disc'].astype('str')
# We do ordinal encoding on the age column

from feature_engine.categorical_encoders import OrdinalCategoricalEncoder



OCE = OrdinalCategoricalEncoder(variables='Age_disc')



OCE.fit(X_train,y_train)



X_train=OCE.transform(X_train)



X_test=OCE.transform(X_test)
# We create a new column with the titles of each passenger

X_train['Title']=X_train['Name'].str.extract('([A-Za-z]+\.)')

X_test['Title']=X_test['Name'].str.extract('([A-Za-z]+\.)')
# We plot the title column to find the rare labels

temp3=X_train['Title'].value_counts()/len(X_train['Title'])



plt.figure(figsize=(12,9))



fig=temp3.sort_values(ascending=False).plot(kind='bar')



fig.axhline(y=0.03,color='red')
# We plot to check if people with a certain title have a higher chance of survival

plt.figure(figsize=(12,9))

sns.countplot(x='Title',hue=y_train,data=X_train)

plt.legend(loc='best')
# We remove the rare labels from the Title Column 



from feature_engine.categorical_encoders import RareLabelCategoricalEncoder



RLC = RareLabelCategoricalEncoder(tol=0.04,n_categories=5,variables='Title')



RLC.fit(X_train)



X_train=RLC.transform(X_train)



X_test=RLC.transform(X_test)
# Convert the title column to string to perform enoding

X_train['Title']=X_train['Title'].astype('str')

X_test['Title']=X_test['Title'].astype('str')
# We now remove the name column from X_train and X_test

X_train=X_train.drop(columns=['Name','Age'])

X_test=X_test.drop(columns=['Name','Age'])
X_train['Title'].value_counts()
from feature_engine.categorical_encoders import OneHotCategoricalEncoder



ohce = OneHotCategoricalEncoder(variables=['Total Family Members','Sex','Embarked','Title','Deck'],drop_last=True)



ohce.fit(X_train)



X_train=ohce.transform(X_train)



X_test=ohce.transform(X_test)
# Convert it into standard scale

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# We use the KNN Classifier to predict the values of X_test

classifier_knn=KNeighborsClassifier()

classifier_knn.fit(X_train,y_train)

y_pred=classifier_knn.predict(X_test)
# now we test the accuracy of the prediction

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

accuracy_knn = classification_report(y_test,y_pred)

cm_knn=confusion_matrix(y_test,y_pred)

accuracy_score_knn=accuracy_score(y_test,y_pred)

print('\n The classification report for KNN Classification is \n{}\n and the accuracy score is {}'.format(accuracy_knn,accuracy_score_knn))
# We use grid search to find the best parameters to use in our classifier algoritham

from sklearn.model_selection import GridSearchCV

parameters_knn=[{'n_neighbors':[1,2,3,4,5],'leaf_size':[10,20,30,40,50],'p':[1,2]}]

grid_search_knn=GridSearchCV(classifier_knn,param_grid=parameters_knn,cv=5,n_jobs=-1,scoring='accuracy')

grid_search_knn=grid_search_knn.fit(X_train,y_train)

grid_search_knn.best_params_
# We use the KNN Classifier after updating the parameters

classifier_knn2=KNeighborsClassifier(leaf_size=10,n_neighbors=3,p=1)

classifier_knn2.fit(X_train,y_train)

y_pred2=classifier_knn2.predict(X_test)
# now we test the accuracy of the new prediction

accuracy_knn2 = classification_report(y_test,y_pred2)

cm_knn2=confusion_matrix(y_test,y_pred2)

accuracy_score_knn2=accuracy_score(y_test,y_pred2)

print('\n The classification report for KNN Classification is \n{}\n and the accuracy score is {}'.format(accuracy_knn2,accuracy_score_knn2))
# We now use the SVC algoritham to build our model

classifier_svc=SVC()



classifier_svc.fit(X_train,y_train)



y_pred_svc=classifier_svc.predict(X_test)



accuracy_svc=classification_report(y_test,y_pred_svc)



cm_svc=confusion_matrix(y_test,y_pred_svc)



accuracy_score_svc=accuracy_score(y_test,y_pred_svc)



print('The classification report for SVC Classification is \n{}\n and the accuracy score is {}'.format(accuracy_svc,accuracy_score_svc))
# We perform grid search on SVC to find out the best parameters for the algoritham

parameters_svc=[{'C':[1,2,3,4,5],'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'gamma':[0.1,0.2,0.3,0.4]}]



grid_search_svc=GridSearchCV(classifier_svc,param_grid=parameters_svc,scoring='accuracy',n_jobs=-1,cv=5)



grid_search_svc=grid_search_svc.fit(X_train,y_train)



grid_search_svc.best_params_
# We Run the SVC algoritham with the updated parameters

classifier_svc2=SVC(C=2,kernel='rbf',gamma=0.2)



classifier_svc2.fit(X_train,y_train)



y_pred_svc2=classifier_svc2.predict(X_test)



accuracy_svc2=classification_report(y_test,y_pred_svc2)



cm_svc2=confusion_matrix(y_test,y_pred_svc2)



accuracy_score_svc2=accuracy_score(y_test,y_pred_svc2)



print('The classification report for SVC Classification is \n{}\n and the accuracy score is {}'.format(accuracy_svc2,accuracy_score_svc2))
# Now we run the Naive Bayes Algoirtham on the dataset

classifier_naive=GaussianNB()



classifier_naive.fit(X_train,y_train)



y_pred_naive=classifier_naive.predict(X_test)



accuracy_naive=classification_report(y_test,y_pred_naive)



cm_naive=confusion_matrix(y_test,y_pred_naive)



accuracy_score_naive=accuracy_score(y_test,y_pred_naive)



print('The classification report for Naive Bayes Classification is \n{}\n and the accuracy score is {}'.format(accuracy_naive,accuracy_score_naive))
# We run the algoritham Random Forest for the dataset

classifier_rf=RandomForestClassifier(random_state=0)



classifier_rf.fit(X_train,y_train)



y_pred_rf=classifier_rf.predict(X_test)



accuracy_rf=classification_report(y_test,y_pred_rf)



cm_rf=confusion_matrix(y_test,y_pred_rf)



accuracy_score_rf=accuracy_score(y_test,y_pred_rf)



print('The classification report for Random Forest Classification is \n{}\n and the accuracy score is {}'.format(accuracy_rf,accuracy_score_rf))
# We use Grid search to find the best parameters



parameters_rf=[{'n_estimators':[1100,1150,1200,1250,1300]}]



grid_search_rf=GridSearchCV(classifier_rf,param_grid=parameters_rf,cv=5,n_jobs=-1,scoring='accuracy')



grid_search_rf=grid_search_rf.fit(X_train,y_train)



grid_search_rf.best_params_
# We run the algoritham with the updated parameters



classifier_rf2=RandomForestClassifier(n_estimators=1100,random_state=0)



classifier_rf2.fit(X_train,y_train)



y_pred_rf2=classifier_rf.predict(X_test)



accuracy_rf2=classification_report(y_test,y_pred_rf2)



cm_rf2=confusion_matrix(y_test,y_pred_rf2)



accuracy_score_rf2=accuracy_score(y_test,y_pred_rf2)



print('The classification report for Random Forest Classification is \n{}\n and the accuracy score is {}'.format(accuracy_rf2,accuracy_score_rf2))
#  We run the gradiant boost algoritham for the dataset

classifier_gb=GradientBoostingClassifier(random_state=0)



classifier_gb.fit(X_train,y_train)



y_pred_gb=classifier_gb.predict(X_test)



accuracy_gb=classification_report(y_test,y_pred_gb)



cm_gb=confusion_matrix(y_test,y_pred_gb)



accuracy_score_gb=accuracy_score(y_test,y_pred_gb)



print('The classification report for Gradient Boost Classification is \n{}\n and the accuracy score is {}'.format(accuracy_gb,accuracy_score_gb))
# We use Grid search to find the best parameters



parameters_gb=[{'n_estimators':[100,200,300,400],'min_samples_split':[2,5,10],'learning_rate':[0.1,0.01,0.001]}]



grid_search_gb=GridSearchCV(classifier_gb,param_grid=parameters_gb,cv=5,n_jobs=-1,scoring='accuracy')



grid_search_gb=grid_search_gb.fit(X_train,y_train)



grid_search_gb.best_params_
# We run the algoritham with the updated parameters



classifier_gb2=GradientBoostingClassifier(learning_rate=0.1,min_impurity_split=10,n_estimators=100,random_state=0)



classifier_gb2.fit(X_train,y_train)



y_pred_gb2=classifier_gb.predict(X_test)



accuracy_gb2=classification_report(y_test,y_pred_gb2)



cm_gb2=confusion_matrix(y_test,y_pred_gb2)



accuracy_score_gb2=accuracy_score(y_test,y_pred_gb2)



print('The classification report for Gradient Boost Classification is \n{}\n and the accuracy score is {}'.format(accuracy_gb2,accuracy_score_gb2))
# we run the model on the Extreme Gradient Boosting Classifier to check if we get a better accuracy

from xgboost import XGBClassifier



classifier_xgb=XGBClassifier(random_state=0)



classifier_xgb.fit(X_train,y_train)



y_pred_xgb=classifier_xgb.predict(X_test)



accuracy_xgb=classification_report(y_test,y_pred_xgb)



cm_xgb=confusion_matrix(y_test,y_pred_xgb)



accuracy_score_xgb=accuracy_score(y_test,y_pred_xgb)



print('The classification report for Extreme Gradient Boost Classification is \n{}\n and the accuracy score is {}'.format(accuracy_xgb,accuracy_score_xgb))
# We check the if the model is overfitting



classifier_xgb1=XGBClassifier(random_state=0)



classifier_xgb1.fit(X_train,y_train)



y_pred_xgb1=classifier_xgb.predict(X_train)



accuracy_xgb1=classification_report(y_train,y_pred_xgb1)



cm_xgb1=confusion_matrix(y_train,y_pred_xgb1)



accuracy_score_xgb1=accuracy_score(y_train,y_pred_xgb1)



print('The classification report for Extreme Gradient Boost Classification is \n{}\n and the accuracy score is {}'.format(accuracy_xgb1,accuracy_score_xgb1))
# We preprocess the test the same way to predict the passenger survival

test=pd.read_csv('/kaggle/input/titanic/test.csv')

X1=test
# We combine the sibilings and parents column to make new column of total family members

X1['Total Family Members']=X1['SibSp']+X1['Parch']+1
X1['Total Family Members']=X1['Total Family Members'].map(family_map)
X1['Deck']=X1['Cabin'].dropna().str[0]



X1['Deck']=X1['Deck'].fillna('M')
# We fill in the missing NaN Values for the fare column

X1['Fare'].fillna(X1['Fare'].mode()[0],inplace=True)
# We fill the NAN values in age column by assigning random values

random_sample_submission=X1['Age'].dropna().sample(X1['Age'].isnull().sum(),random_state=0)



random_sample_submission.index = X1[X1['Age'].isnull()].index



X1.loc[X1['Age'].isnull(), 'Age'] = random_sample_submission
X1.isnull().sum()
# we drop the columns before prediction

X1=X1.drop(columns=['PassengerId','SibSp','Parch','Cabin','Ticket'])
# We bin the age column



X1['Age_disc']=pd.cut(x=X1['Age'],bins=interval,include_lowest=True)



X1['Age_disc']=X1['Age_disc'].astype('str')



# We perform the ordinal encoding in the age_disc column

X1=OCE.transform(X1)
# We discretise the fare column same as in train dataset

X1=ewd.transform(X1)
X1['Title']=X1['Name'].str.extract('([A-Za-z]+\.)')



X1['Title']=X1['Title'].astype('str')
X1
# We perform the rare label encoder

X1=RLC.transform(X1)
# We drop the name and age columns 

X1=X1.drop(columns=['Name','Age'])
# We create dummies for columns which has string elements

X1=ohce.transform(X1)
# We scale the values

X1=sc.transform(X1)
y_pred_test=classifier_knn.predict(X1)
submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission.head(10)
submission['Survived']=y_pred_test
submission.to_csv('Submission.csv',index=False)