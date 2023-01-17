# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head()
train.isnull().sum()
# 性别只有两个类别，所以使用LabelEncoder对性别向量化
from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
encode.fit(train.Sex)
train.Sex=encode.transform(train.Sex)
test.Sex=encode.transform(test.Sex)
train.head()
# 舍弃不别要的特征
drop_cols=['Cabin','PassengerId','Ticket','Embarked']
train.drop(drop_cols,inplace=True,axis=1)
test.drop(drop_cols,inplace=True,axis=1)
train.head()
# 对年龄进行空值填充
from sklearn.impute import SimpleImputer
imput=SimpleImputer()
age_frame=train.Age.to_frame()
test_age_frame=test.Age.to_frame()
imput.fit_transform(age_frame)
train.Age=imput.transform(age_frame)
test.Age=imput.transform(test_age_frame)
train.isnull().sum()
train.Age.describe()
# 对年龄分类
train['age_cat']=pd.cut(train.Age,5,labels=['a_a','a_b','a_c','a_d','a_e'])
test['age_cat']=pd.cut(test.Age,5,labels=['a_a','a_b','a_c','a_d','a_e'])
train.head(10)
train.age_cat.value_counts()
train.Fare.describe()
# 对票价分类
train['fare_cat']=pd.qcut(train.Fare,3,labels=['f_a','f_b','f_c'])
test['fare_cat']=pd.qcut(test.Fare,3,labels=['f_a','f_b','f_c'])
train.fare_cat.value_counts()
# 年龄和票价类别都大于二 所以使用one-hot向量化
f_cats=pd.get_dummies(train['fare_cat'])
a_cats=pd.get_dummies(train['age_cat'])
# 对名字进行处理
def title_classify(data):
    data['Title']=data['Title'].str.replace('Mrs.','123')
    data['Title']=data['Title'].str.replace('Mme.','123')
    data['Title']=data['Title'].str.replace('Mr.','124')
    data['Title']=data['Title'].str.replace('Miss.','125')
    data['Title']=data['Title'].str.replace('Mlle.','125')
    data['Title']=data['Title'].str.replace('Ms.','125')
    data['Title']=data['Title'].str.replace('Master.','126')
    data.loc[data.Title.str.contains('124'),'Title']='Mr'
    data.loc[data.Title.str.contains('123'),'Title']='Mrs'
    data.loc[data.Title.str.contains('125'),'Title']='Miss'
    data.loc[data.Title.str.contains('126'),'Title']='Master'
    data.Title=data.Title.apply(lambda x: 'Rare' if (len(x) > 6) else x)
train['Title']=train['Name'].copy()
title_classify(train)
train.Title.value_counts()
test['Title']=test['Name'].copy()
title_classify(test)
test.Title.value_counts()
# family
train['Family']=train['SibSp']+train['Parch']+1
test['Family']=test['SibSp']+test['Parch']+1
train.Family.value_counts()
train.head()
# 挑选特征 使用one-hot
use_cols=['Pclass','Sex','SibSp','Parch','age_cat','fare_cat','Title','Family']
X=pd.get_dummies(train[use_cols])
X_test=pd.get_dummies(test[use_cols])
y=train['Survived']
print(X.shape)
X.head()
print(X_test.shape)
X_test.head()
from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier()
forest_model.fit(X, y)
forest_pre=forest_model.predict(X_test)
acc_random_forest = round(forest_model.score(X, y) * 100, 2)
acc_random_forest 
from sklearn.ensemble import GradientBoostingClassifier
g_model=GradientBoostingClassifier()
g_model.fit(X,y)
g_pre=g_model.predict(X_test)
acc_g=round(g_model.score(X,y)*100,2)
acc_g
from xgboost import XGBClassifier
xgb_model=XGBClassifier()
xgb_model.fit(X,y)
xgb_pre=xgb_model.predict(X_test)
acc_xgb=round(xgb_model.score(X,y)*100,2)
acc_xgb
