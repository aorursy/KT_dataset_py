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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn import svm



import statsmodels.api as sm



import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)
df = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/merc.csv')

df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()
sns.countplot(df['transmission'])
df['transmission'].value_counts()
df = df[df.transmission != 'Other']

df =df[df.fuelType !='Other']
df['transmission'].value_counts()
sns.countplot(df['transmission'])
sns.countplot(df['fuelType'])
plt.figure(figsize=(15,10))

sns.countplot(df['year'])
sns.pairplot(df)
df['car_age'] = 2020-df['year']

df.head()
sns.barplot(x = 'car_age',y='price',data=df)
sns.barplot(x = 'car_age',y='transmission',data=df)
plt.figure(figsize =(10,10))

sns.countplot(y =df['model'])
sns.countplot(x='model',data=df, order=df.model.value_counts().iloc[:5].index)
plt.figure(figsize =(25,10))

sns.barplot(y = 'price',x='model',data=df)