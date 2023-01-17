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
#导入数据
import pandas
titanic_train = pandas.read_csv("../input/train.csv")
titanic_test = pandas.read_csv("../input/test.csv")
#将性别变成数值
titanic_train_sex = titanic_train["Sex"].unique()
titanic_train.loc[titanic_train["Sex"] == titanic_train_sex[0], "Sex"] = 0
titanic_train.loc[titanic_train["Sex"] == titanic_train_sex[1], "Sex"] = 1
titanic_test_sex = titanic_test["Sex"].unique()
titanic_test.loc[titanic_test["Sex"] == titanic_test_sex[0], "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == titanic_test_sex[1], "Sex"] = 1
#增加姓名长度
titanic_train["NameLength"] = titanic_train["Name"].apply(lambda x: len(x))
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))
#提取，训练数据、训练标签、预测数据
predictors = ["Pclass", "Sex", "Fare", "NameLength"]#给分类器那些特征
train_data = titanic_train.loc[:,predictors]
train_label = titanic_train["Survived"]
test_data = titanic_test.loc[:,predictors]
#填充缺失值
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())
#线性回归预测
from sklearn.linear_model import LinearRegression 
alg = LinearRegression()
alg.fit(train_data, train_label) 
test_predictions = alg.predict(test_data) 
#设置阈值，将预测值映射为类别
test_predictions[test_predictions > .5] = 1
test_predictions[test_predictions <=.5] = 0
#结果输出
pd_data = pandas.DataFrame({'PassengerId':titanic_test["PassengerId"],'Survived':test_predictions})
pd_data.to_csv('Simple_linear.csv',index=False)
print('finish')