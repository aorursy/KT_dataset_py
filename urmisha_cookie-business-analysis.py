# importing libraries

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt 

%matplotlib inline
#importing data

data = pd.read_excel('../input/cookie-business/cookie_business.xlsx')
data.head()
#finding out the shape of the data using "shape" variable: Output (rows, columns)

data.shape
#Printing all the columns present in data

data.columns
#Printing datatypes of different columns

data.dtypes
#To convert Postcode from integer to Object data type

data['Customer ID']=data['Customer ID'].astype(str)
#Printing the updated datatypes

data.dtypes
# list of columns

select_columns = ['Postcode','Age Group', 'Age', 'Gender', 'Favourite Cookie','Cookies bought each week']



# dataframe with specific columns

data[select_columns].head()
# get the summary

data.describe()
#Checking for the missing values in the data

data.isna().sum()
#Calculate the average 

pd.pivot_table(data, index=['Favourite Cookie'], values="Cookies bought each week", aggfunc= ['count'])
#Cross table between Gender and Favourite Cookies

pd.crosstab(data['Gender'], data['Favourite Cookie'])
#Cross table between Gender and Favourite Cookies

pd.crosstab(data['Age Group'], data['Favourite Cookie'])
# Analyze the spread of the "Cookies bought each week" column

sns.distplot(data["Cookies bought each week"], kde = True);
data["Cookies bought each week"].skew()
data.median()
data["Cookies bought each week"].mode()
data.mean()
data["Cookies bought each week"].std()
# What are the different types of Cookies



plot = sns.countplot(x = "Favourite Cookie", data = data)

plot.set_xticklabels(plot.get_xticklabels(), rotation=40);
#Counts the number of cookies

data['Favourite Cookie'].value_counts()
plot = sns.countplot(x = "Age Group", data = data)

plot.set_xticklabels(plot.get_xticklabels(), rotation=40);
#EDA Age Group

data['Age Group'].value_counts()
#Ploting Gender

plot = sns.countplot(x = "Gender", data = data)

plot.set_xticklabels(plot.get_xticklabels(), rotation=40);

#Counts the number of Male and Female buyers

data['Gender'].value_counts()
#Ploting Postcode

plot = sns.countplot(x = "Postcode", data = data)

plot.set_xticklabels(plot.get_xticklabels(), rotation=40);

#Counts Postcodes

data['Postcode'].value_counts()
fig, axs = plt.subplots(figsize = (10,10))

sns.boxplot(x = "Favourite Cookie", y = "Cookies bought each week", data = data, ax=axs, hue='Favourite Cookie');
#Correlation

data["Age"].corr(data["Cookies bought each week"])

data['Postcode'].corr(data['Cookies bought each week'])

data['Age'].corr(data['Postcode'])
data.corr()
corrMatrix = data.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
#Scatterplot between Age and Cookies bought each week

plot = sns.regplot(x = "Age", y = "Cookies bought each week", data = data)
#Scatterplot between Age and Postcode

plot = sns.regplot(x = "Postcode", y = "Age", data = data)
#Scatterplot between Postcode and Cookies bought each week

plot = sns.regplot(x = "Postcode", y = "Cookies bought each week", data = data)
#Bar-plot

fig, axs = plt.subplots(figsize = (15,15))

sns.countplot(x = "Postcode", data = data, ax=axs, hue = "Age Group");
#Bar-plot

fig, axs = plt.subplots(figsize = (15,15))

sns.countplot(x = "Favourite Cookie", data = data, ax=axs, hue = "Age Group");
#Bar-plot

fig, axs = plt.subplots(figsize = (15,15))

sns.countplot(x = "Favourite Cookie", data = data, ax=axs, hue = "Gender");