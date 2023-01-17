import numpy as np

import pandas as pd

import seaborn as sns
train_raw_data=pd.read_csv('../input/titanic/train.csv')

test_raw_data=pd.read_csv('../input/titanic/test.csv')
trainrow=train_raw_data.shape[0]

testrow=test_raw_data.shape[0]

y_train=train_raw_data['Survived'].copy()

train_raw_data=train_raw_data.drop(['Survived'],1)
combine=pd.concat([train_raw_data,test_raw_data])

combine.head()
print(combine.shape)

print(train_raw_data.shape)

print(test_raw_data.shape)
combine.isnull().sum()
#here we can se only three features has missing data

#Emabarked lets replace it with most frequent data

combine['Embarked']=combine['Embarked'].fillna(combine['Embarked'].value_counts().index[0])
combine['Cabin']=combine['Cabin'].fillna('U')

combine['Cabin'].value_counts()

combine['Cabin']=combine['Cabin'].astype(str).str[0]

combine.head()
combine.loc[combine['Fare'].isnull()]
#fare depend on Pclass,Sex, and age

combine['Fare']=combine['Fare'].fillna(combine.loc[(combine['Pclass']==3) & (combine['Sex']=="male") & (combine['Age']<65) & (combine['Age']>55)].dropna()['Fare'].mean())
combine.head()
passengerids=test_raw_data['PassengerId']

combine=combine.drop(['PassengerId','Ticket'],1)
combine.head()
combine['familysize']=combine['SibSp']+combine['Parch']+1

combine.head()
combine['Title'] = combine.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

combine.head()
combine.Title.unique()
combine['Title'].value_counts()
# combine['Title']=combine['Title'].map({"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Major":3,"Col":3,"Mlle":3,"Don":3,"Jonkheer":3,"Countess":3,"Sir":0,"Capt":3,"Mme":2,"Lady":1,"Ms":0,"Dona":3})

# combine.head()
combine=combine.drop(['Name'],1)

combine.head()
combine=combine.drop(['SibSp','Parch'],1)

combine.head()
combine['Sex']=combine['Sex'].map({'male':0,'female':1})

combine.head()
print(combine.Pclass.unique())

print(combine.Sex.unique())
for i in range(0,2):

    for j in range(0,3):

        print(i,j+1)

        temp_dataset=combine[(combine['Sex']==i) &  (combine['Pclass']==j+1)]['Age'].dropna()

        print(temp_dataset)

        #print(str(temp_dataset.median())+"  "+str(i)+"  "+str(j+1))

        combine.loc[(combine.Age.isnull()) & (combine.Sex==i) & (combine.Pclass==j+1),'Age']=int(temp_dataset.median())
combine.isnull().sum()
combine.head()
combine_checkpoint=combine.copy()

combine.head()
combine=combine_checkpoint.copy()

combine.head()
combine['Age_Band']=pd.cut(combine['Age'],5)

combine['Age_Band'].unique()
combine.loc[(combine['Age']<=16.136),'Age']=1

combine.loc[(combine['Age']>16.136) & (combine['Age']<=32.102),'Age']=2

combine.loc[(combine['Age']>32.102) & (combine['Age']<=48.068),'Age']=3

combine.loc[(combine['Age']>48.068) & (combine['Age']<=64.034),'Age']=4

combine.loc[(combine['Age']>64.034) & (combine['Age']<=80.),'Age']=5

combine['Age'].unique()
combine=combine.drop(['Age_Band'],1)
combine['Fare_Band']=pd.cut(combine['Fare'],3)

combine['Fare_Band'].unique()
combine.loc[(combine['Fare']<=170.776),'Fare']=1

combine.loc[(combine['Fare']>170.776) & (combine['Fare']<=314.553),'Fare']=2

combine.loc[(combine['Fare']>314.553) & (combine['Fare']<=513),'Fare']=3

combine=combine.drop(['Fare_Band'],1)
combine['Fare'].value_counts()
combine=pd.get_dummies(columns=['Pclass','Sex','Cabin','Embarked','Title','Age','Fare'],data=combine)

combine.head()
x_train=combine.iloc[:trainrow]

x_test=combine.iloc[trainrow:]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(x_train)

x_scaled_train=scaler.transform(x_train)

x_scaled_train
x_scaled_test=scaler.transform(x_test)

x_scaled_test
from sklearn.linear_model import LogisticRegression

reg=LogisticRegression()

reg.fit(x_scaled_train,y_train)

print(reg.score(x_scaled_train,y_train))

y_pred=reg.predict(x_scaled_test)

y_pred
from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(x_scaled_train,y_train,early_stopping_rounds=5, 

             eval_set=[(x_scaled_train, y_train)], 

             verbose=False)

print(xgb.score(x_scaled_train,y_train))

y_pred=xgb.predict(x_scaled_test)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=4,n_estimators=500,warm_start=True,max_depth=6,min_samples_leaf=2,max_features='sqrt')

rfc.fit(x_scaled_train,y_train)

print(rfc.score(x_scaled_train,y_train))

y_pred=rfc.predict(x_scaled_test)
# from sklearn.neighbors import KNeighborsClassifier

# classifier = KNeighborsClassifier(n_neighbors = 2)

# classifier.fit(x_scaled_train,y_train)

# print(classifier.score(x_scaled_train,y_train))

# y_pred=classifier.predict(x_scaled_test)
passengerids.head()
submission = pd.DataFrame({

        "PassengerId": passengerids,

        "Survived": y_pred

    })

submission
submission.to_csv('submission1.csv', index=False)