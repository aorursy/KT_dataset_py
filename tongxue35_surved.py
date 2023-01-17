# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

#import dataset

pd.set_option('display.max_columns',None)

data_tain=pd.read_csv("../input/train.csv")

data_test=pd.read_csv("../input/test.csv")

print(data_tain.head())

# print(len(data_test))
#knowing the basic info

data_tain.info()

data_tain.describe()

#从中看出各字段类型，值缺失，统计信息

#we can see that about 38.3838% peopel surived ,just calculats datas is number type
#now ploting some figures to kown the data better than codes

fig=plt.figure()

fig.set(alpha=0.3)



plt.subplot2grid((2,3),(0,0))

data_tain.Survived.value_counts().plot.bar()

plt.title('suivived_decs (1 suivived)')

plt.ylabel('people number')



plt.subplot2grid((2,3),(0,1))

data_tain.Pclass.value_counts().plot.bar()



plt.subplot2grid((2,3),(0,2))

plt.scatter(data_tain.Survived,data_tain.Age)    



plt.subplot2grid((2,3),(1,0), colspan=2)

data_tain.Age[data_tain.Pclass == 1].plot(kind='kde')   

data_tain.Age[data_tain.Pclass == 2].plot(kind='kde')

data_tain.Age[data_tain.Pclass == 3].plot(kind='kde')

plt.xlabel('age')

plt.ylabel(u"densty") 

plt.title("all kind of seat distribution")

plt.legend((u'1st', u'2nd',u'3rd'),loc='best')



plt.subplot2grid((2,3),(1,2))

data_tain.Embarked.value_counts().plot.bar()
#calc about sex of survived

S_male=data_tain.Survived[data_tain["Sex"]=='male'].value_counts()

S_female=data_tain.Survived[data_tain["Sex"]=='female'].value_counts()

# df=pd.DataFrame([S_male,S_female])

df=pd.DataFrame({'male':S_male, 'female':S_female})

print(df)

df.plot.bar(stacked=True)

plt.show()

#the lady fist
# info pclass about survived

# two express ways

Survived_0 = data_tain[data_tain['Survived']==0]['Pclass'].value_counts()

Survived_0_1=data_tain.Pclass[data_tain.Survived == 0].value_counts()

Survived_1=data_tain.Pclass[data_tain.Survived == 1].value_counts()

# Survived_0,Survived_1

df1=pd.DataFrame({'survived':Survived_1,'unsurvived':Survived_0})

df1.plot.bar()
#obviously higher Pclass property survived is higher

# feature engineering

# dealing with age.. misssing value





def missing_Age(data):

    age_df=data[["Age","Fare","Parch","SibSp","Pclass"]]

#     print(age_df)

    age_exist=age_df[age_df.Age.notnull()].values

    age_null=age_df[age_df.Age.isnull()].values

    #already known age

    y=age_exist[:,0]

    #feature

    x=age_exist[:,1:]

  

    #trian model

    RFR=RandomForestRegressor(n_estimators=100)

    RFR.fit(x,y)

    predict_age=RFR.predict(age_null[:,1:])

#     print(predict_age)

#     print(data.loc[(data.Age.isnull()),'Age'])

    data.loc[(data.Age.isnull()),'Age']=predict_age

    return data,RFR

data_train,RFR=missing_Age(data_tain)

# data_train

def set_carbin(data):

    data.loc[(data.Cabin.notnull()),'Cabin']='Yes'

    data.loc[(data.Cabin.isnull()),'Cabin']='No'

    return data

data_train=set_carbin(data_train)

data_train.head()



    

    
#encode

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

data_train= pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

data_train.drop(['PassengerId','Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

data_train.head()

# because some columns values ranges different,so need to be standard。



from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# # print(type(data_train[['Age']]))

# # age_scale_param = StandardScaler().fit_transform(data_train)

# age_scale_param = scaler.fit(data_train[['Age']])

# data_train['Age'] = scaler.fit_transform(data_train[['Age']], age_scale_param)

# fare_scaler=scaler.fit(data_train[['Fare']])

# data_train['Fare']=scaler.fit_transform(data_train[['Fare']],fare_scaler)

# sib_scaler=scaler.fit(data_train[['SibSp']])

# data_train['SibSp']=scaler.fit_transform(data_train[['SibSp']],sib_scaler)

# parch_scaler=scaler.fit(data_train[['Parch']])

# data_train['Parch']=scaler.fit_transform(data_train[['Parch']],parch_scaler)

# data_train



def scaler_colums(data):

    scaler=StandardScaler()

    scaler.fit(data)

    data_sc=scaler.fit_transform(data)

    return data_sc

data_train['Age']=scaler_colums(data_train[['Age']])

data_train['Fare']=scaler_colums(data_train[['Fare']])

data_train['Parce']=scaler_colums(data_train[['Parch']])

data_train['SibSp']=scaler_colums(data_train[['SibSp']])

data_train.head()



# build a linear_model.LogisticRegression

from sklearn.linear_model import LogisticRegression

train_data=data_train.values

train_label=train_data[:,0]

train_set=train_data[:,1:]

#no arguments as default

LR=LogisticRegression()

LR.fit(train_set,train_label)

LR
# now here we still need to deal  with the test_data as we did before

# data_test.loc[(data_test.Fare.isnull()),'Fare']

total=data_test.isnull().sum().sort_values(ascending=False)

total

data_test.columns

passengerID=data_test['PassengerId'].values
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0

# data_test.loc[(data_test.Fare.isnull()),'Fare']

test_age_fillna=data_test[["Age","Fare","Parch","SibSp","Pclass"]]

null_age=test_age_fillna[data_test.Age.isnull()].values

predict_age=RFR.predict(null_age[:,1:])

data_test.loc[ (data_test.Age.isnull()), 'Age' ] =predict_age

# data_test.loc[(data_test.Age.isnull()),'Age']

data_test=set_carbin(data_test)

# data_test

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

data_test= pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

data_test.drop(['PassengerId','Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# data_test.head()

data_test['Age']=scaler_colums(data_test[['Age']])

data_test['Fare']=scaler_colums(data_test[['Fare']])

data_test['Parce']=scaler_colums(data_test[['Parch']])

data_test['SibSp']=scaler_colums(data_test[['SibSp']])

data_test.head()
# now predicting...

results=LR.predict(data_test)

final_result = pd.DataFrame({'PassengerId':passengerID, 'Survived':results})

print(final_result)

final_result.to_csv("submission.csv",index=False)
