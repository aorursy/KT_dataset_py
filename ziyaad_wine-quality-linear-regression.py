import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
data = pd.read_csv("../input/winequality-red.csv")



data.head()
data.describe()
data['quality'].value_counts()
data.columns
for i in data.columns[0:11]:

    print(i,stats.spearmanr(data['quality'],data[i]))
data.corr()
x = data[data.columns[0:11]]



y = data['quality']



import seaborn as sns



sns.pairplot(data, x_vars='quality', y_vars=data.columns[0:11], kind='reg')
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
model = LinearRegression()



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.01)



model.fit(x_train,y_train)



a = model.predict(x_test)
from sklearn.metrics import mean_absolute_error



mean_absolute_error(y_test,a)
model.coef_
for i in data.columns[0:11]:

    print(i,stats.spearmanr(data['quality'],data[i]))
print(model.score(x_test,y_test))

print(model.coef_)

fig,ax = plt.subplots()



ax = plt.scatter(y_test,a)



plt.plot()
reg = pd.DataFrame(x_test,columns=x_test.columns)



reg['quality']=a



reg



sns.pairplot(reg,x_vars=reg.columns[0:11],y_vars='quality',kind='reg')



reg['quality']=y_test

sns.pairplot(reg,x_vars=reg.columns[0:11],y_vars='quality',kind='reg')