### Pandas: an open source, 

### BSD-licensed library providing high-performance, 

### easy-to-use data structures and data analysis tools

import pandas as pd



### NumPy is the fundamental package for

### scientific computing with Python

import numpy as np



### Matplotlib is a Python 2D plotting library 

### which produces publication quality figures

### in a variety of hardcopy formats and interactive 

### environments across platforms. 

import matplotlib.pyplot as plt



### Seaborn is a Python visualization library based

### on matplotlib. It provides a high-level interface

### for drawing attractive statistical graphics.

import seaborn as sns



### Seaborn style

### darkgrid, whitegrid, dark, white, ticks

sns.set_style("darkgrid")
### Jupyter prints out the results of the code inline

print("Hello World!!")
3+7
### Let's import our data

train_data = pd.read_csv('../input/train.csv',index_col='PassengerId')

### .head() prints out first 5 rows.

### The Jupyter notebook automatically

### takes the output and makes this nice table

train_data.head()
### This is a straight output of the sum

train_data.isnull().sum()
### Now let's prepare lists of numeric and categorical columns

# Numeric Features

numeric_features = ['Age', 'Fare']

# Categorical Features

ordinal_features = ['Pclass', 'SibSp', 'Parch']

nominal_features = ['Sex', 'Embarked']
### Adding new column with beautiful target names

### This makes maps 0 and 1 to Not Survived and Survived respectively and puts it in a new 

### column

train_data['target_name'] = train_data['Survived'].map({0: 'Not Survived', 1: 'Survived'})
### Target variable exploration

### Seaborn counts Not Survived and Survived

### and makes the plot

sns.countplot(train_data.target_name);



### PLT is the Matplotlib

### Not sure how the sns.countplot gets shown

### without a sns reference being passed into plt

plt.xlabel('Survived?');

plt.ylabel('Number of occurrences');

### This then displays the plot in the Notebook

plt.show()
### If you don't know, Pandas allows you to reference

### columns like train_data['Fare'] or train_data.Fare



### Adjust the size of the plots (Width, Height)

fig = plt.figure(figsize=(10,5))



### SubPlot(RowCount,ColumnCount, Plot Number)

sns.boxplot(train_data['Age'], ax=plt.subplot(121))

sns.boxplot(train_data.Fare, ax=plt.subplot(122))

plt.show()