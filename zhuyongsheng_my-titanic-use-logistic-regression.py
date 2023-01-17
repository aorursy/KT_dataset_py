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

Pclass_WOE=get_woe_data(df_train['Pclass'])

Sex_WOE=get_woe_data(df_train['Sex'])

Age_WOE=get_woe_data(df_train['Age'])

Fare_WOE=get_woe_data(df_train['Fare'])

Embarked_WOE=get_woe_data(df_train['Embarked'])

Title_WOE=get_woe_data(df_train['Title'])

IsAlone_WOE=get_woe_data(df_train['IsAlone'])

AgePclass_WOE=get_woe_data(df_train['AgePclass'])
def get_IV_data(cut,cut_woe):

    grouped=df_train['Survived'].groupby(cut,as_index = True).value_counts()

    cut_IV=((grouped.unstack().iloc[:,1]/df_train["Survived"].sum()-grouped.unstack().iloc[:,0]/(df_train["Survived"].count()-df_train["Survived"].sum()))*cut_woe).sum()    

    return cut_IV



Pclass_IV=get_IV_data(df_train['Pclass'],Pclass_WOE)

Sex_IV=get_IV_data(df_train['Sex'],Sex_WOE)

Age_IV=get_IV_data(df_train['Age'],Age_WOE)

Fare_IV=get_IV_data(df_train['Fare'],Fare_WOE)

Embarked_IV=get_IV_data(df_train['Embarked'],Embarked_WOE)

Title_IV=get_IV_data(df_train['Title'],Title_WOE)

IsAlone_IV=get_IV_data(df_train['IsAlone'],IsAlone_WOE)

AgePclass_IV=get_IV_data(df_train['AgePclass'],AgePclass_WOE)



IV=pd.DataFrame([Pclass_IV,Sex_IV,Age_IV,Fare_IV,Embarked_IV,Title_IV,IsAlone_IV,AgePclass_IV],index=['Pclass_IV','Sex_IV','Age_IV','Fare_IV','Embarked_IV','Title_IV','IsAlone_IV','AgePclass_IV'],columns=['IV'])

iv=IV.plot.bar(color='b',alpha=0.3,rot=30,figsize=(10,5),fontsize=(10))

iv.set_title('自变量与IV值分布图',fontsize=(15))

iv.set_xlabel('自变量',fontsize=(15))

iv.set_ylabel('IV',fontsize=(15))

IV
import seaborn as sns

corr=df_train.corr()

fig=plt.figure(figsize=(9,9))

ax1=fig.add_subplot(1,1,1)

sns.heatmap(corr,annot=True,cmap='YlGnBu',ax=ax1,annot_kws={'size':12,'weight':'bold','color':'blue'})
df['Pclass']=df['Pclass']-1

df_train['Pclass']=df_train['Pclass']-1

#df['AgePclass']=df['AgePclass']-1

#df_train['AgePclass']=df_train['AgePclass']-1

df=df.drop(['AgePclass'],axis=1)

#df=df.drop(['Age'],axis=1)

#df=df.drop(['Sex'],axis=1)

df_train=df_train.drop(['AgePclass'],axis=1)

#df_train=df_train.drop(['Age'],axis=1)

#df_train=df_train.drop(['Sex'],axis=1)
df_train.describe()
df.head()
for each in ['Embarked','Fare','Pclass','Title','IsAlone','Age','Sex']:

    df[each]=df[each].apply(lambda x:eval(each+'_WOE').iloc[x])
df_train=df[df['Survived'].notnull()]

df_test=df[df['Survived'].isnull()]

x_train = df_train[['Embarked','Fare','Pclass','Title','IsAlone','Age','Sex']]

y_train = df_train['Survived']

model=LogisticRegressionCV(cv=3,random_state=0)

model.fit(x_train,y_train)
df_test.head()
print(model.score(x_train,y_train))
df_test.isnull().sum()
y_pred=model.predict(df_test[['Embarked','Fare','Pclass','Title','IsAlone','Age','Sex']])
y_pred=y_pred.astype(int)
y_pred
Submission = pd.DataFrame({ 'PassengerId': test_PassengerId,

                            'Survived': y_pred })

Submission.to_csv("Submission.csv", index=False)
Submission