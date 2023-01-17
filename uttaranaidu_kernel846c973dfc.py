import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/StudentsPerformance.csv")
data.head()
data.sample(5)
data.info()
data.describe()
data['gender'].value_counts
sns.countplot(x='gender',data=data)

plt.ylabel('frequence')

plt.title('Gender Bar Plot')

plt.show()
sns.countplot(x='parental level of education',data=data)

plt.xticks(rotation=45)

plt.ylabel('frequence')

plt.title('Parental level of education Bar Plot')

plt.show()
sns.barplot(x='gender',y='reading score',data=data)

plt.show()
sns.barplot(x='gender',y='reading score',hue='race/ethnicity',data=data)

plt.show()
sns.distplot(data['math score'],bins =10,kde=True)
sns.jointplot(x='math score',y='writing score',data=data)
sns.pairplot(data)
sns.boxplot(x='gender',y='math score',data=data)

plt.show()
sns.boxplot(x='gender',y='writing score',data=data)

plt.show()
sns.boxplot(x='race/ethnicity',y='writing score',data=data)

plt.show()
data.corr()
sns.heatmap(data.corr())
g=sns.catplot(x='gender',y='writing score',hue='lunch',col_wrap=3,col='race/ethnicity',data=data,kind='bar',height=4,aspect=0.7)

plt.show()