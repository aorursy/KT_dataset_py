# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
len(train)
train.describe()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train.info()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
import matplotlib.pyplot as plt

#encoding=utf-8

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = train.Pclass[train.Survived == 0].value_counts()

Survived_1 = train.Pclass[train.Survived == 1].value_counts()

df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"各乘客等级的获救情况")

plt.xlabel(u"乘客等级") 

plt.ylabel(u"人数") 

plt.show()
fig = plt.figure()

fig.set(alpha=0.5)



survived_0 = train.Sex[train.Survived == 0].value_counts()

survived_1 = train.Sex[train.Survived == 1].value_counts()

df=pd.DataFrame({u'survived':survived_1, u'not survived':survived_0})

df.plot(kind='bar', stacked=True)

plt.title('sex difference')

plt.xlabel('sex')

plt.ylabel('count')

plt.show()
fig = plt.figure()

fig1 = fig.add_subplot(131)

train.Survived[train.Embarked == 'C'].value_counts().plot(kind='bar', label="embarked C", color='#FF0000')

fig1.set_xticklabels([u"survived", u"not survived"], rotation=0)

plt.legend("C")



fig2 = fig.add_subplot(132)

train.Survived[train.Embarked == 'Q'].value_counts().plot(kind='bar', label="embarked Q", color='#00FF00')

fig2.set_xticklabels([u"survived", u"not survived"], rotation=0)

plt.legend("Q")



fig3 = fig.add_subplot(133)

train.Survived[train.Embarked == 'S'].value_counts().plot(kind='bar', label="embarked Q", color='#0000FF')

fig3.set_xticklabels([u"survived", u"not survived"], rotation=0)

plt.legend("S")



plt.show()
dummies_Embarked = pd.get_dummies(train['Embarked'], prefix= 'Embarked')



dummies_Sex = pd.get_dummies(train['Sex'], prefix= 'Sex')



dummies_Pclass = pd.get_dummies(train['Pclass'], prefix= 'Pclass')



df = pd.concat([train, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df
import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()

# age_scale_param = scaler.fit(df['Age'])

# df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)

fare_scaled = scaler.fit_transform(df['Fare'].values.reshape(-1,1))

df['Fare_scaled'] = fare_scaled

df
from sklearn import linear_model

train_df = df.filter(regex='Survived|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')

train_np = train_df.as_matrix()

# y即Survival结果

y = train_np[:, 0]



# features

x = train_np[:,1:]



model = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

model.fit(x, y)

model
dummies_Embarked = pd.get_dummies(test['Embarked'], prefix= 'Embarked')



dummies_Sex = pd.get_dummies(test['Sex'], prefix= 'Sex')



dummies_Pclass = pd.get_dummies(test['Pclass'], prefix= 'Pclass')



df_test = pd.concat([test, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test
df_test['Fare'].fillna(0,inplace=True)

df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1))


df_test.describe()
df_test
test_data = df_test.filter(regex='Survived|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = model.predict(test_data)

result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})

result
result.to_csv('titanic.csv',index=False)