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
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv', index_col = 'id')
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
train.columns
nom_cols = ['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7']
X = train.drop('target', axis=1)
y = train.target
for col in nom_cols:
    print("=======" + col + "=======")
    print(train[col].value_counts())
    print("unique values : " + str(len(train[col].unique())))
plt.figure(figsize = (18,8))
sns.barplot(x = train.nom_0, y=y)
plt.figure(figsize = (18,8))
sns.barplot(x=train.nom_1, y=y)
plt.figure(figsize = (18,8))
sns.barplot(x=train.nom_2, y=y)
plt.figure(figsize = (18,8))
sns.barplot(x=train.nom_3, y=y)
plt.figure(figsize=(18,8))
sns.barplot(x=train.nom_4, y=y)
nom_col = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
for i in range(5):
    for j in range(5):
        plt.figure(figsize=(18,5))
        sns.barplot(x=train[nom_col[i]], y=y, hue=train[nom_col[j]])
train.groupby(['nom_0', 'nom_1', 'nom_4']).target.mean()
len(train.nom_5.unique())
plt.figure(figsize=(18,8))
sns.barplot(x=train.nom_5, y=y)
sns.lineplot(x=train.nom_5, y=y.mean())
plt.figure(figsize=(18,8))
sns.distplot(train.groupby('nom_5').target.mean())
dict(train.groupby('nom_5').target.std())

nom_5_dict = dict(train.groupby('nom_5').target.mean())
nom_5_dict
for key in list(nom_5_dict.keys()):
    if nom_5_dict[key] < 0.2:
        nom_5_dict[key] = 'a'
    elif nom_5_dict[key] >= 0.2 and nom_5_dict[key] < 0.25:
        nom_5_dict[key] = 'b'
    elif nom_5_dict[key] >= 0.25 and nom_5_dict[key] < 0.3:
        nom_5_dict[key] = 'c'
    elif nom_5_dict[key] >= 0.3 and nom_5_dict[key] <0.35:
        nom_5_dict[key] = 'd'
    elif nom_5_dict[key] >= 0.35 and nom_5_dict[key] < 0.4:
        nom_5_dict[key] = 'e'
    elif nom_5_dict[key] >=0.4:
        nom_5_dict[key] = 'f'
train.nom_5.replace(nom_5_dict).value_counts()
plt.figure(figsize=(18,8))
sns.barplot(x=train.nom_5.replace(nom_5_dict), y=y)
len(train.nom_6.unique())
plt.figure(figsize=(18,8))
sns.distplot(train.groupby('nom_6').target.mean())
len(train.nom_7.unique())
plt.figure(figsize=(18,8))
sns.distplot(train.groupby('nom_7').target.mean())
plt.figure(figsize=(18,8))
sns.distplot(train.groupby('nom_8').target.mean())
plt.figure(figsize=(18,8))
sns.distplot(train.groupby('nom_9').target.mean())