import numpy as np

import pandas as pd

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score,StratifiedKFold, learning_curve





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

       print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/train.csv')

df_test=pd.read_csv('/kaggle/input/test.csv')

test_PassengerId=df_test['PassengerId']

df=df.append(df_test)
df[df.Survived.isnull()].head()
df = df.drop(['Ticket', 'Cabin'], axis=1)
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace('Ms', 'Miss')

df['Title'] = df['Title'].replace('Mme', 'Mrs')

    

title_mapping={'Mr':0,'Rare':1,'Master':2,'Miss':3,'Mrs':4}

df['Title']=df['Title'].map(title_mapping)

df['Title']=df['Title'].fillna(0)

df.head()
df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
guess_Ages=np.zeros((2,3))

guess_Ages
for i in range(0,2):

    for j in range(0,3):

        guess_df=df[(df['Sex']==i)&(df['Pclass']==j+1)]['Age'].dropna()

        Age_guess=guess_df.median()

        guess_Ages[i,j] = int( Age_guess/0.5 + 0.5 ) * 0.5



for i in range(0,2):

    for j in range(0,3):

        df.loc[(df.Age.isnull())&(df['Sex']==i)&(df['Pclass']==j+1),'Age']=guess_Ages[i,j]

    

df['Age']=df['Age'].astype(int)
df['AgeBand']=pd.cut(df['Age'],5)
df.loc[ df['Age'] <= 16, 'Age'] = 0

df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

df.loc[ df['Age'] > 64, 'Age']=4
df=df.drop(['PassengerId','Name'],axis=1)
df= df.drop(['AgeBand'], axis=1)
df['FamilySize']=df['SibSp']+df['Parch']+1
df['IsAlone']=0

df.loc[(df['FamilySize']==1),'IsAlone']=1
df = df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
df['AgePclass']=df.Age*df.Pclass
freq_port=df.Embarked.dropna().mode()[0]
df['Embarked'] = df['Embarked'].fillna(freq_port)
#why not C:0,Q:1,S:2?

df['Embarked']=df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)
df['FareBand'] = pd.qcut(df['Fare'], 4)
df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0

df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2

df.loc[ df['Fare'] > 31, 'Fare'] = 3

df['Fare'] = df['Fare'].astype(int)



df = df.drop(['FareBand'], axis=1)    
df_train=df[df['Survived'].notnull()]

rate=df_train['Survived'].sum()/(df_train['Survived'].count()-df_train['Survived'].sum())
def get_woe_data(cut):

    grouped=df_train['Survived'].groupby(cut,as_index = True).value_counts()

    woe=np.log(grouped.unstack().iloc[:,1]/grouped.unstack().iloc[:,0]/rate)

    return woe

import seaborn as sns

corr=df_train.corr()

fig=plt.figure(figsize=(9,9))

ax1=fig.add_subplot(1,1,1)

sns.heatmap(corr,annot=True,cmap='YlGnBu',ax=ax1,annot_kws={'size':12,'weight':'bold','color':'blue'})
df['Pclass']=df['Pclass']-1

df_train['Pclass']=df_train['Pclass']-1

#df['AgePclass']=df['AgePclass']-1

#df_train['AgePclass']=df_train['AgePclass']-1

#df=df.drop(['AgePclass'],axis=1)

#df=df.drop(['Age'],axis=1)

#df=df.drop(['Sex'],axis=1)

#df_train=df_train.drop(['AgePclass'],axis=1)

#df_train=df_train.drop(['Age'],axis=1)

#df_train=df_train.drop(['Sex'],axis=1)
df_train.describe()
df.head()
df_train=df[df['Survived'].notnull()]

df_test=df[df['Survived'].isnull()]

x_train = df_train[['Embarked','Fare','Pclass','Title','IsAlone','Age','AgePclass','Sex']]

#x_train = df_train[['Embarked','Fare','Pclass','Title','IsAlone','Age']]

y_train = df_train['Survived']

model=LogisticRegressionCV(cv=5,random_state=0)

model.fit(x_train,y_train)
df_test.head()
print(model.score(x_train,y_train))
x_test= df_test[['Embarked','Fare','Pclass','Title','IsAlone','Age','AgePclass','Sex']]

#x_test= df_test[['Embarked','Fare','Pclass','Title','IsAlone','Age']]
df_test.isnull().sum()
y_pred=model.predict(df_test[['Embarked','Fare','Pclass','Title','IsAlone','Age','AgePclass','Sex']])
y_pred=y_pred.astype(int)
kfold = StratifiedKFold(n_splits=5)



random_state = 2
SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(x_train,y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
SVMC_y_pred=gsSVMC.predict(df_test[['Embarked','Fare','Pclass','Title','IsAlone','Age','AgePclass','Sex']])
SVMC_y_pred=SVMC_y_pred.astype(int)
RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 8],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(x_train,y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
RF_y_pred=gsRFC.predict(df_test[['Embarked','Fare','Pclass','Title','IsAlone','Age','AgePclass','Sex']])

RF_y_pred=RF_y_pred.astype(int)
np.corrcoef(y_pred,RF_y_pred)
np.corrcoef(SVMC_y_pred,RF_y_pred)
T_y_pred=y_pred+RF_y_pred+SVMC_y_pred
for i in range(len(T_y_pred)):

    if T_y_pred[i]<2:

        T_y_pred[i]=0

    else:

        T_y_pred[i]=1
T_y_pred
Submission = pd.DataFrame({ 'PassengerId': test_PassengerId,

                            'Survived': T_y_pred })

Submission.to_csv("Submission.csv", index=False)
Submission