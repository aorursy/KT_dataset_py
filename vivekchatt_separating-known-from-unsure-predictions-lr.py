# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

data = pd.read_csv('../input/train.csv')
data.head()
plt.figure(figsize=(12,7))

sns.heatmap(data.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
plt.figure(figsize=(12,7))

sns.countplot(data['Pclass'],hue=data['Survived'])
plt.figure(figsize=(12,7))

sns.boxplot(y=data['Age'],x=data['Survived'])
plt.figure(figsize=(12,7))

sns.boxplot(y=data['Age'],x=data['Pclass'])
def impute_Age(val):

    Age = val[0]  

    Pclass  = val[1]  

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else: return 23

    else: return Age

    

   
data['Age'] = data[['Age','Pclass']].apply(impute_Age,axis=1)
plt.figure(figsize=(12,7))

sns.heatmap(data.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
Train = data.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)
Train.head()    
Sex = pd.get_dummies(data['Sex'],drop_first=True)
Embarked = pd.get_dummies(data['Embarked'],drop_first=True)
Train = pd.concat([Train,Sex,Embarked],axis=1)
Train.drop(['Sex','Embarked'],axis=1,inplace=True)
Train.head()
Train.head()
def can_Sacrifice(cols):

    Parch = cols[0]

    SibSp = cols[1]

    Survived = cols[2]

    if SibSp != 0:

        if Parch != 0:

            if Survived == 0:

                return 1        

            else: return 0            

        elif Survived == 0:

            return 1        

        else: return 0        

    elif Survived == 0:

         return 1    

    else: return 0

        
Train['Sacrifice'] = Train[['Parch','SibSp','Survived']].apply(can_Sacrifice,axis=1)
Train.corr()
def Alone(var):

    Parch = var[0]

    SibSp = var[1]

    

    if Parch == 0:

        if SibSp==0:

            return 1

        else: return 0

    else: return 0    
Train['Alone'] = Train[['Parch','SibSp']].apply(Alone,axis=1)
def sur_Chance(vart):

    Sacrifice = vart[0]

    Alone = vart[1]

    male = vart[2]

    Pclass = vart[3]

    

    if Sacrifice == 1:

        if Alone == 1:

            if male == 1:

                if Pclass == 1:

                    return 'Unsure'

                elif Pclass == 3:

                    return 'Low'

                else: return 'Unsure'

            else: return 'Unsure'

        else: return 'Low'

    else: return 'High'
Train['Survival'] = Train[['Sacrifice','Alone','male','Pclass']].apply(sur_Chance,axis=1)
plt.figure(figsize=(12,7))

sns.countplot(Train['Survival'])
Train.corr()
Train.drop('Survived',axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
LE.fit(Train['Survival'])
LE.classes_
Train['Survival'] = LE.transform(Train['Survival'])
X = Train.drop('Survival',axis=1)
y = Train['Survival']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
sns.pairplot(Train)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs',multi_class='multinomial',C = 0.1)
fit = lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test,y_pred)

cr = classification_report(y_test,y_pred)
print(cm)

print(cr)
plt.figure(figsize=(12,7))

sns.countplot(y_pred)
plt.figure(figsize=(12,7))

sns.countplot(y_test)