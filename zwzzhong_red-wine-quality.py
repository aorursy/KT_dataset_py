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
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns
from collections import Counter
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
wine = pd.read_csv('../input/winequality-red.csv')
wine.head()
wine.describe()
wine.info()
wine['alcohol_level'] = wine['alcohol'].apply(lambda x: 'high' if x > 10.5 else 'low')
wine['chlorides_level'] = wine['chlorides'].apply(lambda x: 'high' if x > 0.09 else 'low')

def qua(x):
    if(x > 6):
        return 'high'
    elif(x > 4):
        return 'mid'
    else:
        return 'low' 


wine['quality_level'] = wine['quality'].apply(qua)

wine['pH_level'] = wine['pH'].apply(lambda x: 'strong' if x > 3.3 else 'mild')

wine.head()
plt.figure(figsize=(20,15))
sns.countplot(x='quality',data=wine)
plt.figure(figsize=(20,15))
sns.countplot(x='quality_level',data=wine)
plt.figure(figsize=(20,15))
wine['pH_level_1'] = pd.cut(wine['pH'], bins = 10)
sns.boxplot(x='pH_level_1', y='quality', data=wine)
plt.figure(figsize=(20,15))
wine['alcohol_level_1'] = pd.cut(wine['alcohol'], bins = 10)
sns.boxplot(x='alcohol_level_1', y='quality', data=wine)
w = sns.FacetGrid(wine, col='quality', hue='pH_level', col_wrap=2, height=7)
w.map(plt.scatter,'chlorides','alcohol')
plt.legend()
plt.figure(figsize=(20,15))
sns.pairplot(wine)
plt.figure(figsize=(20,15))
wine_corr = wine.corr()
mask = np.zeros_like(wine_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(round(wine_corr,2), mask=mask, annot=True, linewidths=2)
wine.info()
wine = wine.drop(['alcohol_level', 'chlorides_level', 'quality_level', 'pH_level', 'alcohol_level_1', 'pH_level_1'], axis = 1)
X= wine.iloc[:,:-1].values
y= wine.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder
labelencoder_y =LabelEncoder()
y= labelencoder_y.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)
wine.head()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_test
X_train