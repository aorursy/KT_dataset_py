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
# importing some plotting tools:

import matplotlib.pyplot as plt

import seaborn as sns



# importing tools for preprocessing the data:

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# importing the algorithms needed to predict:

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor





from keras.losses import mean_absolute_percentage_error as mape
data = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")

data.head()
y = data['Chance of Admit ']



data.drop(['Chance of Admit ', 'Serial No.'], inplace=True, axis=1)

plt.figure(figsize=(8,8))

sns.set_palette(['paleturquoise', 'deepskyblue'])

sns.set_context("poster", font_scale=0.7)

sns.scatterplot(data=data, x='GRE Score', y='TOEFL Score', hue='Research')
sns.set_palette(['m'])

plt.figure(figsize=(8,8))

sns.distplot(data['University Rating'])
sns.set_palette(['mediumpurple'])

plt.figure(figsize=(8,8))

sns.distplot(data['SOP'])
plt.figure(figsize=(8,8))

sns.violinplot(data['CGPA'])
xtrain, xtest, ytrain, ytest = train_test_split(data, y, train_size=0.5, test_size=0.5)
RanModel = RandomForestRegressor(n_estimators=500)



RanModel.fit(xtrain, ytrain)
XGModel = XGBRegressor(n_estimators=500)



XGModel.fit(xtrain, ytrain)
RanPreds = RanModel.predict(xtest)



mape(ytest, RanPreds)
XGPreds = XGModel.predict(xtest)

mape(ytest, XGPreds)
for i in range(0, 10):

    print(RanModel.predict(xtest)[i])
print(ytest[0:10])