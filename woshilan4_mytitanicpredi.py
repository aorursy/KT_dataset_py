import pandas as pd

trainFile='../input/titanic/train.csv'

testFile='../input/titanic/test.csv'

titanic1=pd.read_csv(trainFile)

titanic2=pd.read_csv(testFile)

titanic=titanic1.append(titanic2)

titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())#fillna（）表示补充，median（）表示求平均值

titanic['Fare']=titanic['Fare'].fillna(titanic['Fare'].median())

print(titanic.describe())
print(titanic['Sex'].unique())
titanic.loc[titanic['Sex']=='male','Sex']=0

titanic.loc[titanic['Sex']=='female','Sex']=1
print(titanic['Survived'].unique())

titanic['Survived']=titanic['Survived'].fillna(0)

print(titanic['Survived'].unique())


print(titanic['Embarked'].unique())

titanic['Embarked']=titanic['Embarked'].fillna('S')

titanic.loc[titanic["Embarked"]=="S","Embarked"] = 0

titanic.loc[titanic["Embarked"]=="C","Embarked"] = 1

titanic.loc[titanic["Embarked"]=="Q","Embarked"] = 2

print(titanic['Embarked'].unique())
print(titanic['Pclass'].unique())
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(18,6), dpi=1600)

alpha=alpha_scatterplot = 0.2

alpha_bar_chart = 0.55

# 让我们一起绘制许多不同形状的图

ax1 = plt.subplot2grid((2,3),(0,0))

# 绘制出那些生存与没有生存的人的条形图

titanic.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)

ax1.set_xlim(-1, 2)

# 在我们的图表上放置一个标题

plt.title("Distribution of Survival, (1 = Survived)")

plt.subplot2grid((2,3),(0,1))

plt.scatter(titanic.Survived, titanic.Age, alpha=alpha_scatterplot)

# sets the y axis lable

plt.ylabel("Age")

# 绘制图形二

# 格式化图形的网格线样式

plt.grid(b=True, which='major', axis='y')

plt.title("Survival by Age, (1 = Survived)")

ax3 = plt.subplot2grid((2,3),(0,2))

titanic.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)

ax3.set_ylim(-1, len(titanic.Pclass.value_counts()))

plt.title("Class Distribution")

plt.subplot2grid((2,3),(1,0), colspan=2)

#绘制了一等通行人年龄的子集的核密度估计

titanic.Age[titanic.Pclass == 1].plot(kind='kde')

titanic.Age[titanic.Pclass == 2].plot(kind='kde')

titanic.Age[titanic.Pclass == 3].plot(kind='kde')

# 绘制轴标签

plt.xlabel("Age")

plt.title("Age Distribution within classes")

# sssss为我们的图表设置我们的图例

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')

ax5 = plt.subplot2grid((2,3),(1,2))

titanic.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)

ax5.set_xlim(-1, len(titanic.Embarked.value_counts()))

#指定我们图的参数

plt.title("Passengers per boarding location")
#使用回归算法(二分类)进行预测

#线性回归

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold

#交叉验证:将训练数据集分成3份，对这三份进行交叉验证，比如使用1，2样本测试，3号样本验证

#对最后得到得数据取平均值



#选中一些特征

predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

alg = LinearRegression()

#n_folds代表将数据切分成3份，存在3层的交叉验证，titanic.shape[0]代表样本个数

kf = KFold(n_splits=3,shuffle=False, random_state=1)



predictions = []

for train,test in kf.split(titanic):

    #iloc通过行号获取数据

    train_predictors = titanic[predictors].iloc[train,:]

    #获取对应的label值

    train_target = titanic["Survived"].iloc[train]

    #进行训练

    alg.fit(train_predictors,train_target)

    #进行预测

    test_predictors = alg.predict(titanic[predictors].iloc[test,:])

    #将结果加入到list中

    predictions.append(test_predictors)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

 

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

 

# Initialize our algorithm with the default paramters

# n_estimators is the number of trees we want to make

# min_samples_split is the minimum number of rows we need to make a split

# min_samples_leaf is the minimum number of samples we can have at the place where a

# tree branch(分支) ends (the bottom points of the tree)

alg = RandomForestClassifier(random_state=1,

                             n_estimators=10,

                             min_samples_split=2,

                             min_samples_leaf=1)

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)

kf = KFold(n_splits=3, shuffle=False, random_state=1)

scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

 

# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
import numpy as np

 

# The predictions are in three separate numpy arrays. Concatenate them into one.

# We concatenate them on asix 0, as they only have one axis.

predictions = np.concatenate(predictions,axis=0)

 

# Map predictions to outcomes (only possible outcomes are 1 and 0)

predictions[predictions > .5] = 1  # 映射成分类结果 计算准确率

predictions[predictions <= .5] = 0

 

# 注意这一行与源代码有出入

accuracy = sum(predictions==titanic['Survived'])/len(predictions)

 

# 验证集的准确率

print(accuracy)