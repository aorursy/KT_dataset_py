# import libraries

import pandas as pd

import numpy as np
automobile = pd.read_csv('../input/Automobile_data.txt')

automobile.head()
automobile.dtypes
automobile.describe()
automobile.isnull().sum()
# Find out number of records having '?' value for normalized losses

automobile['normalized-losses'].loc[automobile['normalized-losses'] == '?'].count()
# Setting the missing value to mean of normalized losses and conver the datatype to integer

nl = automobile['normalized-losses'].loc[automobile['normalized-losses'] != '?']

nlmean = nl.astype(str).astype(int).mean()

automobile['normalized-losses'] = automobile['normalized-losses'].replace('?',nlmean).astype(int)

automobile['normalized-losses'].head()
# Find out the number of values which are not numeric

automobile['price'].str.isnumeric().value_counts()
# List out the values which are not numeric

automobile['price'].loc[automobile['price'].str.isnumeric() == False]
#Setting the missing value to mean of price and convert the datatype to integer

price = automobile['price'].loc[automobile['price'] != '?']

pmean = price.astype(str).astype(int).mean()

automobile['price'] = automobile['price'].replace('?',pmean).astype(int)

automobile['price'].head()
# Checking the numberic and replacing with mean value and conver the datatype to integer

automobile['horsepower'].str.isnumeric().value_counts()

horsepower = automobile['horsepower'].loc[automobile['horsepower'] != '?']

hpmean = horsepower.astype(str).astype(int).mean()

automobile['horsepower'] = automobile['horsepower'].replace('?',pmean).astype(int)
#Checking the outlier of horsepower

automobile.loc[automobile['horsepower'] > 10000]
#Excluding the outlier data for horsepower

automobile[np.abs(automobile.horsepower-automobile.horsepower.mean())<=(3*automobile.horsepower.std())]
# Find out the number of invalid value

automobile['bore'].loc[automobile['bore'] == '?']
# Replace the non-numeric value to null and conver the datatype

automobile['bore'] = pd.to_numeric(automobile['bore'],errors='coerce')

automobile.dtypes
# Replace the non-number value to null and convert the datatype

automobile['stroke'] = pd.to_numeric(automobile['stroke'],errors='coerce')

automobile.dtypes
# Convert the non-numeric data to null and convert the datatype

automobile['peak-rpm'] = pd.to_numeric(automobile['peak-rpm'],errors='coerce')

automobile.dtypes
# remove the records which are having the value '?'

automobile['num-of-doors'].loc[automobile['num-of-doors'] == '?']

automobile = automobile[automobile['num-of-doors'] != '?']

automobile['num-of-doors'].loc[automobile['num-of-doors'] == '?']
import matplotlib.pyplot as plt

% matplotlib inline
automobile.make.value_counts().nlargest(10).plot(kind='bar', figsize=(15,5))

plt.title("Number of vehicles by make")

plt.ylabel('Number of vehicles')

plt.xlabel('Make');
automobile.symboling.hist(bins=6,color='green');

plt.title("Insurance risk ratings of vehicles")

plt.ylabel('Number of vehicles')

plt.xlabel('Risk rating');
automobile['normalized-losses'].hist(bins=5,color='orange');

plt.title("Normalized losses of vehicles")

plt.ylabel('Number of vehicles')

plt.xlabel('Normalized losses');
automobile['fuel-type'].value_counts().plot(kind='bar',color='purple')

plt.title("Fuel type frequence diagram")

plt.ylabel('Number of vehicles')

plt.xlabel('Fuel type');
automobile['aspiration'].value_counts().plot.pie(figsize=(6, 6), autopct='%.2f')

plt.title("Fuel type pie diagram")

plt.ylabel('Number of vehicles')

plt.xlabel('Fuel type');
automobile.horsepower[np.abs(automobile.horsepower-automobile.horsepower.mean())<=(3*automobile.horsepower.std())].hist(bins=5,color='red');

plt.title("Horse power histogram")

plt.ylabel('Number of vehicles')

plt.xlabel('Horse power');
automobile['curb-weight'].hist(bins=5,color='brown');

plt.title("Curb weight histogram")

plt.ylabel('Number of vehicles')

plt.xlabel('Curb weight');
automobile['drive-wheels'].value_counts().plot(kind='bar',color='grey')

plt.title("Drive wheels diagram")

plt.ylabel('Number of vehicles')

plt.xlabel('Drive wheels');
automobile['num-of-doors'].value_counts().plot(kind='bar',color='purple')

plt.title("Number of doors frequency diagram")

plt.ylabel('Number of vehicles')

plt.xlabel('Number of doors');
import seaborn as sns

corr = automobile.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

a = sns.heatmap(corr, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
plt.rcParams['figure.figsize']=(23,10)

ax = sns.boxplot(x="make", y="price", data=automobile)
g = sns.lmplot('price',"engine-size", automobile);
g = sns.lmplot('normalized-losses',"symboling", automobile);
plt.scatter(automobile['engine-size'],automobile['peak-rpm'])

plt.xlabel('Engine size')

plt.ylabel('Peak RPM');
g = sns.lmplot('city-mpg',"curb-weight", automobile, hue="make", fit_reg=False);
g = sns.lmplot('highway-mpg',"curb-weight", automobile, hue="make",fit_reg=False);
automobile.groupby('drive-wheels')['city-mpg'].mean().plot(kind='bar', color = 'peru');

plt.title("Drive wheels City MPG")

plt.ylabel('City MPG')

plt.xlabel('Drive wheels');
automobile.groupby('drive-wheels')['highway-mpg'].mean().plot(kind='bar', color = 'peru');

plt.title("Drive wheels Highway MPG")

plt.ylabel('Highway MPG')

plt.xlabel('Drive wheels');
plt.rcParams['figure.figsize']=(10,5)

ax = sns.boxplot(x="drive-wheels", y="price", data=automobile)
pd.pivot_table(automobile,index=['body-style','num-of-doors'], values='normalized-losses').plot(kind='bar',color='purple')

plt.title("Normalized losses based on body style and no. of doors")

plt.ylabel('Normalized losses')

plt.xlabel('Body style and No. of doors');