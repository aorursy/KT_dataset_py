import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/data-classified/classified_data.csv',index_col=0)

df.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])

df_feat.head()
from sklearn.model_selection import train_test_split



X=df_feat

y=df['TARGET CLASS']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred= knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
error_rate=[]

for i in range(300,320):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i=knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(300,320),error_rate,color='blue',linestyle='--',marker='o',markerfacecolor='red',markersize=10)

plt.title('error rate vs k value')

plt.xlabel('k')

plt.ylabel('error rate')

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=310)

knn.fit(X_train,y_train)

pred=knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(pred,y_test))

print(classification_report(pred,y_test))
####################################################################################################################
#Titanic applying KNN
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')
def impute(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age



train['Age']=train[['Age','Pclass']].apply(impute,axis=1)
train.drop('Cabin',axis=1,inplace=True)
sex=pd.get_dummies(train['Sex'],drop_first=True)

embark=pd.get_dummies(train['Embarked'],drop_first=True)

train=pd.concat([train,sex,embark],axis=1)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
def impute1(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 42

        elif Pclass==2:

            return 26

        else:

            return 23

    else:

        return Age



test['Age']=test[['Age','Pclass']].apply(impute1,axis=1)
test.drop('Cabin',axis=1,inplace=True)

test['Fare'].loc[152]=np.mean(test['Fare'])
sex=pd.get_dummies(test['Sex'],drop_first=True)

embark=pd.get_dummies(test['Embarked'],drop_first=True)

test=pd.concat([test,sex,embark],axis=1)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()



scaler.fit(train.drop('Survived',axis=1))

scaled_features1 = scaler.transform(train.drop('Survived',axis=1))



scaler.fit(test)

scaled_features2 = scaler.transform(test)
train_feat=pd.DataFrame(scaled_features1,columns=train.columns.drop('Survived'))

test_feat=pd.DataFrame(scaled_features2,columns=test.columns)
X=train_feat

y=train['Survived']
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X,y)

pred=knn.predict(test_feat)
def createList(r1,r2):

    return [item for item in range(r1,r2)]

PassengerId=createList(892,1310)

PassengerId=pd.DataFrame({'':PassengerId}).rename(columns={'':'PassengerId'})

PassengerId

Survived=pd.DataFrame({'':list(pred)})
Survived=pd.DataFrame({'':list(pred)}).rename(columns={'':'Survived'})

Survived
result=pd.concat([PassengerId,Survived],axis=1)
result.to_csv('titanic_mark1.csv',index=False)