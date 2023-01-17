# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.model_selection import cross_val_score

import numpy 

from sklearn.preprocessing import LabelEncoder

from sklearn import neighbors

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import linear_model
iris = pd.read_csv('/kaggle/input/iris.data',header=None)

iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']

le = LabelEncoder()

le.fit(iris['Species'])

features =  ['SepalWidthCm','PetalWidthCm']

X = iris[features]

y = le.transform(iris['Species'])
knn = neighbors.KNeighborsClassifier(10,weights='uniform')

model = knn.fit(X,y)

score = numpy.mean(cross_val_score(knn,X,y,cv=5,scoring='accuracy'))

print('平均性能得分：'+str(score))
Forecast = knn.predict(X)#用原来的X，利用knn模型进行预测

Forecast = pd.DataFrame(Forecast)#转为pandas数据框的格式



iris = pd.merge(iris,pd.DataFrame(y),how='inner',right_index=True,left_index=True)#把离散化的新的分类变量合并到iris

iris = pd.merge(iris,Forecast,how='inner',right_index=True,left_index=True)#把预测的离散分类变量也合并到iris



iris.rename(columns={ iris.columns[5]: "Species_NO" }, inplace=True)#对新的列进行重命名

iris.rename(columns={ iris.columns[6]: "Species_NO_Forecast" }, inplace=True)#对新的列进行重命名



sns.relplot(x="SepalWidthCm", y="PetalWidthCm", hue="Species_NO", palette="Set1",data=iris)#把分类在图表上画出来。

sns.relplot(x="SepalWidthCm", y="PetalWidthCm", hue="Species_NO_Forecast", palette="Set1",data=iris)#把预测分类分类都在图表上画出来。
knn.predict([[3,1.4]]) #新数据预测SepalWidthCm=3,PetalWidthCm=1.4进行预测
lm = linear_model.LogisticRegression()

model = lm.fit(X,y)

score = numpy.mean(cross_val_score(lm,X,y,cv=5,scoring='accuracy'))

print('平均性能得分：'+str(score))