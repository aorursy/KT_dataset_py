# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# importing matplotlib and Seaborn libraries
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline

#Import the dataset
data = pd.read_csv(r'/kaggle/input/insurance/insurance.csv')
data.head()
data.shape
rows=data.shape[0]
columns=data.shape[1]
print('Total Rows :-',rows)
print('Total Columns :- ',columns)
data.dtypes
data.isnull().sum()
data.describe().transpose()
f, axes = plt.subplots(1, 3,figsize=(15,5))
BMI = sns.distplot(data['bmi'], color="green", kde=True,ax=axes[0])
AGE = sns.distplot(data['age'], color="blue", kde=True,ax=axes[1])
CHARGES = sns.distplot(data['charges'], color="red", kde=True,ax=axes[2])
dataFrame = pd.DataFrame(data)
skewness = dataFrame.skew(axis=0) # axis=0 for column
print('Skewness of ‘bmi’, ‘age’ and ‘charges’ columns')
print(skewness)
f, axes = plt.subplots(3, 1, figsize=(15, 15))
bmiBoxPlot = sns.boxplot(data['bmi'], color="yellow",ax=axes[0])
ageBoxPlot = sns.boxplot(data['age'], color="blue",ax=axes[1])
chargesBoxPlot = sns.boxplot(data['charges'], color="red",ax=axes[2])

list(data.columns.values)
# ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
# categorical columns :- sex,children,smoker,region

f, axes = plt.subplots(2, 2, figsize=(15, 15))
sexCountPlot = sns.countplot(data = data, x = 'sex',ax=axes[0,0])
childrenCountPlot = sns.countplot(data = data, x = 'children',ax=axes[0,1])
smokerCountPlot = sns.countplot(data = data, x = 'smoker',ax=axes[1,0])
regionCountPlot = sns.countplot(data = data, x = 'region',ax=axes[1,1])

sns.pairplot(data,hue = 'smoker')
sns.pairplot(data, hue='sex')
sns.pairplot(data, hue='children')
sns.pairplot(data, hue='region')