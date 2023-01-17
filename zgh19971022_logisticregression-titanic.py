import numpy as np 
import pandas as pd 
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
data_train = pd.read_csv('../input/titanic/train.csv')
data_train
data_train.info()
#可以看出有些值是不全的
data_train.describe()
#从均值上可以看出一些东西的
fig = plt.figure()
fig.set(alpha = 0.2)

plt.subplot2grid((2,3), (0,0))
data_train.Survived.value_counts().plot(kind = 'bar')
plt.title(u"surv")
plt.ylabel(u"number")

plt.subplot2grid((2,3), (0,1))
data_train.Pclass.value_counts().plot(kind = "bar")
plt.ylabel(u"number")
plt.title(u"level")
fig = plt.figure()
fig.set(alpha = 0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'survived':Survived_1, u'no survived':Survived_0})
df.plot(kind = 'bar', stacked = True)
plt.title(u"level result")
plt.xlabel(u"level")
plt.ylabel(u"number")
plt.show()

#地位比较重要

fig = plt.figure()
fig.set(alpha = 0.2)

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u'male':Survived_m, u'female':Survived_f})
df.plot(kind = 'bar', stacked = True)
plt.title(u"Sex result")
plt.xlabel(u"Sex")
plt.ylabel(u"number")
plt.show()

data_train.Cabin.value_counts()
##feature engineering

##用随机森林填补缺失
def set_missing_ages(df):
    #丢
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    #分成已知年龄和未知年龄
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    
    y = known_age[:, 0]
    X = known_age[:, 1:]
    
    rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs =-1)
    rfr.fit(X, y)
    #预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    #填补
    df.loc[ (df.Age.isnull()), 'Age'] = predictedAges
    
    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()),'Cabin'] = "Yes"
    df.loc[ (df.Cabin.isnull()),'Cabin'] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
data_train.info()
#因子化
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix = 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix = 'Embarked')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix = 'Pclass')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix = 'Sex')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis = 1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
df.head()

#scaling(特征化到[-1,1])
scalar = preprocessing.StandardScaler()
age_scale_param = scalar.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scalar.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)
fare_scale_param = scalar.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scalar.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)
df.head()
#正则抽取feature
train_df = df.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values
                     
y = train_np[:, 0]

X = train_np[:, 1:]

clf = linear_model.LogisticRegression(solver = 'liblinear',C = 1.0, penalty = 'l1',tol = 1e-6)
clf.fit(X,y)
clf
#回头来预处理test
data_test = pd.read_csv('../input/titanic/test.csv')
data_test.loc[ (data_test.Fare.isnull()), 'Fare'] = 0
#特征变换、随机森林
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
#补全年龄
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()),'Age'] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix = 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix = 'Embarked')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix = 'Pclass')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix = 'Sex')


df_test = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis = 1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
df_test

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis = 1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
df_test
df_test['Age_scaled'] = scalar.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scalar.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)
df_test.head()
test = df_test.filter(regex = 'Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
output = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
output.to_csv('ZGH_LR_submission.csv', index = False)
pd.read_csv("./ZGH_LR_submission.csv").head()
#Bagging模型融合
train_df = df.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values
                     
y = train_np[:, 0]

X = train_np[:, 1:]

clf = linear_model.LogisticRegression(solver = 'liblinear',C = 1.0, penalty = 'l1',tol = 1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators = 20, max_samples = 0.8, max_features = 1.0, bootstrap = True, bootstrap_features = False, n_jobs =-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex = 'Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = bagging_clf.predict(test)
output = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
output.to_csv('ZGH_LR_submission2.csv', index = False)
#交叉检验
all_data = df.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
                     
y = all_data.values[:, 0]

X = all_data.values[:, 1:]

print(cross_val_score(clf, X, y, cv = 5))
