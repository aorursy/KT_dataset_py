import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
titanic = pd.read_csv("../input/train.csv")
titanic.head()
titanic.shape
titanic.info()
titanic_test = pd.read_csv("../input/test.csv")
titanic_test.head()
titanic.describe()
titanic.describe(include=['O'])
titanic.isnull().sum()
titanic_test.info()
titanic_test.isnull().sum()
print("Proportion of Survived:",sum(titanic['Survived']==1)/len(titanic['Survived']))
print("Proportion of NOT Survived:",sum(titanic['Survived']==0)/len(titanic['Survived']))
titanic['Sex'].value_counts()
titanic.groupby('Sex').Survived.value_counts()
titanic['Pclass'].value_counts()
titanic.groupby('Pclass').Survived.value_counts()
titanic[['Pclass','Survived']].groupby('Pclass',as_index=False).mean()
titanic[['Sex','Survived']].groupby('Sex',as_index=False).mean()
sns.barplot('Sex','Survived',data=titanic)
sns.barplot('Pclass','Survived',data=titanic)
titanic['Embarked'].value_counts()
titanic.groupby('Embarked').Survived.value_counts()
titanic[['Embarked','Survived']].groupby('Embarked',as_index=False).mean()
sns.barplot('Embarked','Survived',data=titanic)
titanic['Parch'].value_counts()
titanic.groupby('Parch').Survived.value_counts()
titanic[['Parch','Survived']].groupby('Parch',as_index=False).mean()
sns.barplot('Parch','Survived',data=titanic,ci=None)
titanic['SibSp'].value_counts()
titanic.groupby('SibSp').Survived.value_counts()
titanic[['SibSp','Survived']].groupby('SibSp',as_index=False).mean()
sns.barplot('SibSp','Survived',data=titanic)
plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
sns.violinplot('Sex','Age',data=titanic,hue='Survived',split=True)
plt.subplot(1,3,2)
sns.violinplot('Pclass','Age',data=titanic,hue='Survived',split=True)
plt.subplot(1,3,3)
sns.violinplot('Embarked','Age',data=titanic,hue='Survived',split=True)
train_test_df = [titanic,titanic_test]
for df in train_test_df:
    df['Sex'] = df['Sex'].map({'female':1,'male':0})
titanic.head()
titanic_test.head()
titanic['Embarked'].unique()
titanic['Embarked'].value_counts()
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic['Embarked'].unique()
for df in train_test_df:
    df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2})
titanic.head()
sns.boxplot(titanic['Age'])
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean()).astype(int)
titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].mean()).astype(int)
for df in train_test_df:
    df.loc[(df['Age']<=10),'Age'] = 0
    df.loc[(df['Age']>10) & (df['Age']<=21) ,'Age'] = 1
    df.loc[(df['Age']>21) & (df['Age']<=35) ,'Age'] = 2
    df.loc[(df['Age']>35) & (df['Age']<=50) ,'Age'] = 3
    df.loc[(df['Age']>50) & (df['Age']<=65) ,'Age'] = 4
    df.loc[(df['Age']>65),'Age'] = 5
titanic.head()
sns.boxplot(titanic['Fare'])
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median()).astype(int)
titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median()).astype(int)
for df in train_test_df:
    df.loc[(df['Fare']<=2),'Fare'] = 0
    df.loc[(df['Fare']>2) & (df['Fare']<=5) ,'Fare'] = 2
    df.loc[(df['Fare']>5) & (df['Fare']<=8) ,'Fare'] = 3
    df.loc[(df['Fare']>8) & (df['Fare']<=15) ,'Fare'] = 4
    df.loc[(df['Fare']>15) & (df['Fare']<=50) ,'Fare'] = 5
    df.loc[(df['Fare']>50),'Fare'] = 6
titanic.head()
for df in train_test_df:
    df['Title'] = df['Name'].str.extract('([A-Za-z]+\.)')
titanic.head()
titanic['Title'].unique()
titanic_test.Title.unique()
titanic['Title'] = titanic['Title'].replace(['Don.','Dona.','Rev.','Dr.','Major.','Lady.', 'Sir.', 'Col.', 'Capt.','Countess.', 'Jonkheer.'],'Other')
titanic['Title'] = titanic['Title'].replace('Ms.','Miss.')
titanic['Title'] = titanic['Title'].replace('Mlle.','Miss.')
titanic['Title'] = titanic['Title'].replace('Mme.','Miss.')
titanic_test['Title'].unique()
titanic_test['Title'] = titanic_test['Title'].replace(['Don.','Dona.','Rev.','Dr.','Major.','Lady.', 'Sir.', 'Col.', 'Capt.','Countess.', 'Jonkheer.'],'Other')
titanic_test['Title'] = titanic_test['Title'].replace('Ms.','Miss.')
titanic_test['Title'] = titanic_test['Title'].replace('Mlle.','Miss.')
titanic_test['Title'] = titanic_test['Title'].replace('Mme.','Miss.')
titanic_test['Title'].unique()
for df in train_test_df:
    df['Title'] = df['Title'].map({'Mr.':0,'Mrs.':1,'Miss.':2,'Master.':3,'Other':4})
titanic_test.isnull().sum()
titanic.head()
for df in train_test_df:
    df['FamilySize'] = df['SibSp'] +  df['Parch'] + 1
titanic.isnull().sum()
#titanic['Cabin'] = titanic['Cabin'].fillna(titanic['Cabin'].mode()[0])
#titanic_test['Cabin'] = titanic_test['Cabin'].fillna(titanic_test['Cabin'].mode()[0])
titanic['Cabin'] = titanic['Cabin'].fillna('Missing')
titanic['Cabin'] = titanic['Cabin'].str[:1]
titanic_test['Cabin'] = titanic_test['Cabin'].fillna('Missing')
titanic_test['Cabin'] = titanic_test['Cabin'].str[:1]
titanic.head()
titanic['Cabin'].unique()
features_drop = ['Name', 'Ticket']
titanic = titanic.drop(features_drop, axis=1)
titanic_test = titanic_test.drop(features_drop, axis=1)
titanic = titanic.drop(['PassengerId'], axis=1)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
t_cabin = le.fit_transform(titanic['Cabin'])
te_cabin = le.fit_transform(titanic_test['Cabin'])
t_cabin = pd.DataFrame(t_cabin,columns=['Cabins'])
te_cabin = pd.DataFrame(te_cabin,columns=['Cabins'])
t_cabin.head()
titanic = pd.concat([titanic,t_cabin],axis=1)
titanic.head()
titanic_test = pd.concat([titanic_test,te_cabin],axis=1)
titanic_test.head()
features_drop = ['Cabin']
titanic = titanic.drop(features_drop, axis=1)
titanic_test = titanic_test.drop(features_drop, axis=1)
titanic.tail()
titanic.groupby(['Survived','Age'])['Age'].count()
titanic.info()
titanic_test.info()
titanic_test.head()
titanic_test.info()
feature1 = ['Pclass','Sex','Cabins','FamilySize']
feature2 = ['Age','SibSp','Fare']
feature3 = ['Parch','Embarked','Title']
X_train = titanic.drop('Survived',axis=1)
y_train = titanic['Survived']
X_test = titanic_test.drop('PassengerId',axis=1).copy()

X_test.isnull().sum()
X_train1 = X_train[feature1+feature3]
X_train2 = X_train[feature2+feature3]
X_train3 = X_train[feature1+feature2]
X_test1 = X_test[feature1+feature3]
X_test2 = X_test[feature2+feature3]
X_test3 = X_test[feature1+feature2]
print(X_train.shape)   
print(y_train.shape)    
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
score = lr.score(X_train, y_train)
score
from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)
score = svc.score(X_train, y_train)
score
lnsvc = LinearSVC()
lnsvc.fit(X_train,y_train)
y_pred_lnsvc = lnsvc.predict(X_test)
score = lnsvc.score(X_train, y_train)
score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred_dtc = dtc.predict(X_test)
score = dtc.score(X_train, y_train)
score
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
y_pred_rfc = rfc.predict(X_test)
score = rfc.score(X_train, y_train)
score
rfc3 = RandomForestClassifier(n_estimators=100)
rfc3.fit(X_train3,y_train)
y_pred_rfc3 = rfc3.predict(X_test3)
score3 = rfc3.score(X_train3, y_train)
score3
import xgboost
classifier1 = xgboost.XGBClassifier()
classifier1.fit(X_train1, y_train)
# Predicting the Test set results
y_pred_XGB1 = classifier1.predict(X_test1)
acc_XGB1 = round( classifier1.score(X_train1, y_train) * 100, 2)
print ("Train Accuracy: " + str(acc_XGB1) + '%')
y_pred_XGB1.shape
import xgboost
classifier2 = xgboost.XGBClassifier()
classifier2.fit(X_train2, y_train)
# Predicting the Test set results
y_pred_XGB2 = classifier2.predict(X_test2)
acc_XGB2 = round( classifier2.score(X_train2, y_train) * 100, 2)
print ("Train Accuracy: " + str(acc_XGB2) + '%')
y_pred_XGB2.shape
import xgboost
classifier3 = xgboost.XGBClassifier()
classifier3.fit(X_train3, y_train)
# Predicting the Test set results
y_pred_XGB3 = classifier3.predict(X_test3)
acc_XGB3 = round( classifier3.score(X_train3, y_train) * 100, 2)
print ("Train Accuracy: " + str(acc_XGB3) + '%')
y_pred_XGB3.shape
import lightgbm as lgb
train_data3=lgb.Dataset(X_train3,label=y_train)
#define parameters
params = {'n_estimators':200,'objective':'binary','learning_rate':0.1,'max_depth': 10,'num_leaves':'100','min_data_in_leaf':9,'max_bin':100,'boosting_type':'gbdt',}
model3= lgb.train(params, train_data3, 200) 
y_pred_LGB3=model3.predict(X_test3)
#rounding the values
y_pred_LGB3=y_pred_LGB3.round(0)
#converting from float to integer
y_pred_LGB3=y_pred_LGB3.astype(int)

import lightgbm as lgb
train_data2=lgb.Dataset(X_train2,label=y_train)
#define parameters
params = {'n_estimators':200,'objective':'binary','learning_rate':0.1,'max_depth': 10,'num_leaves':'100','min_data_in_leaf':9,'max_bin':100,'boosting_type':'gbdt',}
model2= lgb.train(params, train_data2, 200) 
y_pred_LGB2=model2.predict(X_test2)
#rounding the values
y_pred_LGB2=y_pred_LGB2.round(0)
#converting from float to integer
y_pred_LGB2=y_pred_LGB2.astype(int)
import lightgbm as lgb
train_data1=lgb.Dataset(X_train1,label=y_train)
#define parameters
params = {'n_estimators':200,'objective':'binary','learning_rate':0.1,'max_depth': 10,'num_leaves':'100','min_data_in_leaf':9,'max_bin':100,'boosting_type':'gbdt',}
model1= lgb.train(params, train_data1, 200) 
y_pred_LGB1=model1.predict(X_test1)
#rounding the values
y_pred_LGB1=y_pred_LGB1.round(0)
#converting from float to integer
y_pred_LGB1=y_pred_LGB1.astype(int)
Mix_XGB = (y_pred_LGB1 +  y_pred_LGB2 + y_pred_LGB3)/3
Mix_XGB = pd.DataFrame(Mix_XGB,columns=['Survived'])
#rounding the values
#Mix_XGB=Mix_XGB.round(0)
#converting from float to integer
#Mix_XGB=Mix_XGB.astype(int)
Mix_XGB['Survived'] = Mix_XGB['Survived'].apply(lambda x: 1 if x > 0.5 else 0)

submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": Mix_XGB['Survived']
    })

submission.to_csv('gender_submission.csv', index=False)
submission.Survived.value_counts()
# votingC = VotingClassifier(estimators=[('LGR', lr), ('SVC', svc),('LGB_1',model1),('LGB_2',model2),('LGB_3',model3)], voting='soft', n_jobs=4)

# votingC = votingC.fit(X_train, y_train)
votingC = VotingClassifier(estimators=[('LGR', lr),
 ('DTC',dtc),('RFC',rfc),('RFC_3',rfc3),('XGB_1',classifier1),('XGB_2',classifier2),('XGB_3',classifier3)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, y_train)
vote = votingC.predict(X_test)
submission_v = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": vote
    })

submission_v.to_csv('Vote_submission.csv', index=False)
submission_v.Survived.value_counts()




