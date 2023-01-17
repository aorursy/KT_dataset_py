import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline
train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.isnull()
sns.heatmap(train.isnull(),yticklabels=False)
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',data=train,hue="Sex")
sns.countplot(x='Survived',data=train,hue="Pclass")
sns.distplot(train['Age'].dropna(),kde=False)
sns.countplot(x='SibSp',data=train)
sns.distplot(train['Fare'].dropna(),kde=False)
plt.figure(figsize=(15,7))
sns.boxplot(x='Pclass',y='Age',data=train)
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False)
train.drop('Cabin',axis=1,inplace=True)
train
sns.heatmap(train.isnull(),yticklabels=False)
train['Embarked'].value_counts()
sex=pd.get_dummies(train['Sex'],drop_first=True)
embarked=pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train
train = pd.concat([train,sex,embarked,],axis=1)
train.head()
train.drop('Survived',axis=1)
train['Survived'].head()
