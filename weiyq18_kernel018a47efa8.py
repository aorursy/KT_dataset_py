# -*- coding: utf-8 -*-

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
import pandas as pd #数据分析

import numpy as np #科学计算

from pandas import Series,DataFrame

data_train = pd.read_csv("../input/train.csv")

data_train.info()

data_train.describe()
import matplotlib.pyplot as plt

fig = plt.figure()

fig.set(alpha=0.2)



plt.subplot2grid((2,3),(0,0))

data_train.Survived.value_counts().plot(kind='bar')

plt.title(u'获救情况（1为获救）')

plt.ylabel(u'number')



plt.subplot2grid((2,3),(0,1))

data_train.Pclass.value_counts().plot(kind="bar")

plt.ylabel(u"人数")

plt.title(u"乘客等级分布")



plt.subplot2grid((2,3),(0,2))

plt.scatter(data_train.Survived, data_train.Age)

plt.ylabel(u"年龄")                         # 设定纵坐标名称

plt.grid(b=True, which='major', axis='y') 

plt.title(u"按年龄看获救分布 (1为获救)")



plt.subplot2grid((2,3),(1,0), colspan=2)

data_train.Age[data_train.Pclass == 1].plot(kind='kde')   

data_train.Age[data_train.Pclass == 2].plot(kind='kde')

data_train.Age[data_train.Pclass == 3].plot(kind='kde')

plt.xlabel(u"年龄")# plots an axis lable

plt.ylabel(u"密度") 

plt.title(u"各等级的乘客年龄分布")

plt.legend((u'1等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.



plt.subplot2grid((2,3),(1,2))

data_train.Embarked.value_counts().plot(kind='bar')

plt.title(u"各登船口岸上船人数")

plt.ylabel(u"人数")  

plt.show()
# -*- coding: utf-8 -*-

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()

df=pd.DataFrame({u'saved':Survived_1, u'died':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"各乘客等级的获救情况")

plt.xlabel(u"乘客等级") 

plt.ylabel(u"人数") 

plt.show()

fig = plt.figure()

fig.set(alpha=0.2)



Survived_m = data_train.Survived[data_train.Sex=='male'].value_counts()

Survived_f = data_train.Survived[data_train.Sex=='female'].value_counts()

df=pd.DataFrame({'male':Survived_m,'female':Survived_f})

df.plot(kind='bar',stacked=True)

plt.title('按性别看获救情况')

plt.xlabel('gander')

plt.ylabel('number')

plt.show()

 #然后我们再来看看各种舱级别情况下各性别的获救情况

fig=plt.figure()

fig.set(alpha=0.65) # 设置图像透明度，无所谓

plt.title(u"根据舱等级和性别的获救情况")



ax1=fig.add_subplot(141)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')

ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)

ax1.legend([u"女性/高级舱"], loc='best')



ax2=fig.add_subplot(142, sharey=ax1)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')

ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)

plt.legend([u"女性/低级舱"], loc='best')



ax3=fig.add_subplot(143, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')

ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)

plt.legend([u"男性/高级舱"], loc='best')



ax4=fig.add_subplot(144, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')

ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)

plt.legend([u"男性/低级舱"], loc='best')



plt.show()

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()

df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"各登录港口乘客的获救情况")

plt.xlabel(u"登录港口") 

plt.ylabel(u"人数") 



plt.show()

from sklearn.ensemble import RandomForestRegressor



def set_missing_ages(df):

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    

    known_age= age_df[age_df.Age.notnull()].as_matrix()

    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    

    y = known_age[:,0]

    X = known_age[:,1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测

    predictedAges = rfr.predict(unknown_age[:, 1::])

    

    # 用得到的预测结果填补原缺失数据

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df, rfr



def set_Cabin_type(df):

    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"

    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"

    return df



data_train, rfr = set_missing_ages(data_train)

data_train = set_Cabin_type(data_train)

data_train

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')



dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')



dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')



dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')



df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df

import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))

df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))

df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)

df



from sklearn import linear_model

from sklearn.ensemble import BaggingRegressor



# 用正则取出我们要的属性值

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

train_np = train_df.as_matrix()

y = train_np[:, 0]

X = train_np[:, 1:]

# fit到RandomForestRegressor之中

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

bagging_clf = BaggingRegressor(clf,n_estimators=20,max_samples=0.8,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=-1)

bagging_clf.fit(X,y)

# clf.fit(X, y)

# clf
data_test = pd.read_csv("../input/test.csv")

data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0

# 接着我们对test_data做和train_data中一致的特征变换

# 首先用同样的RandomForestRegressor模型填上丢失的年龄

tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

null_age = tmp_df[data_test.Age.isnull()].as_matrix()

# 根据特征属性X预测年龄并补上

X = null_age[:, 1:]

predictedAges = rfr.predict(X)

data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges



data_test = set_Cabin_type(data_test)

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')





df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)

df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)





test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = bagging_clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})

result.to_csv("submission.csv", index=False)



result = pd.read_csv("submission.csv")

# result
from sklearn.model_selection import cross_val_score,train_test_split



 #简单看看打分情况

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

X = all_data.as_matrix()[:,1:]

y = all_data.as_matrix()[:,0]

print (cross_val_score(clf, X, y, cv=5))

from sklearn.model_selection import cross_val_score,train_test_split

split_train, split_cv = train_test_split(df, test_size=0.3, random_state=0)

train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')



# 生成模型

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])



# 对cross validation数据进行预测



cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = clf.predict(cv_df.as_matrix()[:,1:])



origin_data_train = pd.read_csv("../input/train.csv")

bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]

# bad_cases

pd.DataFrame({"columns":list(train_df.columns)[1:],"coef":list(clf.coef_.T)})