# 所需要的套件

import pandas as pd # data processing

from matplotlib import pyplot as plt # data visualization
# 读取档案

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# 观看数据的前几行

train.head()
# 观察数据缺失

train.isnull().sum()
# 补值

train['Age'] = train['Age'].fillna(train['Age'].median())
# 补值，转换栏位，变成有无客舱

train['Cabin'][~train['Cabin'].isnull()] = 1

train['Cabin'][train['Cabin'].isnull()] = 0
# 观察登船口岸人数

train.groupby('Embarked')['PassengerId'].count()
# 补值，转换栏位数值

train["Embarked"] = train["Embarked"].fillna("S")

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2
# 转换栏位数值

train["Sex"][train["Sex"] == "male"] = 1

train["Sex"][train["Sex"] == "female"] = 0
# 观看数据的前几行

train.head()
# 观察数据缺失

train.isnull().sum()
# 观察数据统计

train.describe(include='all')
# 观察数据统计

train.shape
# 生存数

print(train["Survived"].value_counts())

# 百分比

print(train["Survived"].value_counts(normalize = True))
# 性别生存比较

train['Died'] = 1 - train['Survived']

train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True)
# 年龄生存比较

figure = plt.figure(figsize=(25, 7))

plt.hist([train[train['Survived'] == 1]['Age'], train[train['Survived'] == 0]['Age']],

         stacked=True, bins=50, label=['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()
# 票价生存比较

figure = plt.figure(figsize=(25, 7))

plt.hist([train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']],

         stacked=True, bins=50, label=['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()
# 机器学习模型测试（非完整机器学习流程）

# 完整流程 -> 小小AI发名家

# 需要的套件

from sklearn import tree



# 从训练数据中撷取目标向量与输入向量

y = train["Survived"].values

X = train[["Pclass", "Sex", "Age", "Fare", "Parch", "Cabin", "Embarked"]].values



# 训练

dt = tree.DecisionTreeClassifier()

dt = dt.fit(X, y)



# 训练结果

print(dt.feature_importances_)

print(dt.score(X, y))
# 观察测试数据

test.head()
# 观察测试数据缺失

test.isnull().sum()
# 补值，转换栏位

test["Fare"] = test["Fare"].fillna(train["Fare"].median())

test["Age"] = test["Age"].fillna(train["Age"].median())



test["Sex"][test["Sex"] == "male"] = 1

test["Sex"][test["Sex"] == "female"] = 0



test['Cabin'][~test['Cabin'].isnull()] = 1

test['Cabin'][test['Cabin'].isnull()] = 0



test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2
# 从测试数据中撷取输入向量

test_features = test[["Pclass", "Sex", "Age", "Fare", "Parch", "Cabin", "Embarked"]].values



# 测试

my_prediction = dt.predict(test_features)

print(my_prediction)



# 建立结果数据

import numpy as np

PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_solution)



# 观察结果

print(my_solution.shape)



# 储存档案

my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])
# 观察储存档案

!ls