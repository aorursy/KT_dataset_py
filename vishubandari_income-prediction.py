# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/adult-income-dataset/adult.csv")
data.head()
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel
from scipy import stats
data.info()
data_num = data.copy()
data.head()
attrib, counts = np.unique(data['workclass'], return_counts = True)
print(attrib,counts)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
print(most_freq_attrib)
data['workclass'][data['workclass'] == '?'] = most_freq_attrib 


attrib, counts = np.unique(data['occupation'], return_counts = True)
print(attrib,counts)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
print(most_freq_attrib)
data['occupation'][data['occupation'] == '?'] = most_freq_attrib 


attrib, counts = np.unique(data['native-country'], return_counts = True)
print(attrib,counts)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
print(most_freq_attrib)
data['native-country'][data['native-country'] == '?'] = most_freq_attrib 


data.head()
for id in data.columns:
    print(data[id].unique())

a1 = data['income'][0]
a2 = data['income'][2]
print(a1)
data['income'] = data['income'].map({'<=50K': 0, '>50K' : 1 }, na_action='ignore' )
data.head()
data.income.unique()
data_num = data.drop(['educational-num', 'income'], axis = 1)  #as income and educational-num distribution has no significance
data_num.describe()
data.describe(include=["O"])
data['age'].hist(figsize=(8,8))
plt.show()
data[data['age']>70].shape
# print(data['age'])
data['hours-per-week'].hist(figsize=(8,8))
plt.show()
data['fnlwgt'].hist(figsize=(8,8))
plt.show()
data['capital-gain'].hist(figsize=(8,8))
plt.show()
data["capital-loss"].hist(figsize=(8,8))
plt.show()
data[data["capital-loss"]>0].shape
import seaborn as sns
plt.figure(figsize=(12,8))

total = float(len(data["income"]) )

ax = sns.countplot(x="workclass", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()
plt.figure(figsize=(12,8))

total = float(len(data["education"]) )

ax = sns.countplot(x="education", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()
plt.figure(figsize=(15,8))
total = float(len(data) )

ax = sns.countplot(x="marital-status", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()
plt.figure(figsize=(30,10))
total = float(len(data) )

ax = sns.countplot(x="occupation", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()
plt.figure(figsize=(15,8))
total = float(len(data) )

ax = sns.countplot(x="relationship", data=data)
# for p in ax.patches:
#     height = p.get_height()
#     ax.text(p.get_x()+p.get_width()/2.,
#             height + 3,
#             '{:1.2f}'.format((height/total)*100),
#             ha="center") 
plt.show()
total = float(len(data) )

ax = sns.countplot(x="race", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()
ax = sns.countplot(x="gender", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()
ax = sns.countplot(y="native-country", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()
ax = sns.countplot(x="income", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()
plt.figure(figsize=(10,10)) 
sns.boxplot(x="income", y="age", data=data)
plt.show()
data[['income','age']].groupby(['income'], as_index=False).mean()
data.columns()
