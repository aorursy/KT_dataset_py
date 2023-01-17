import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/black-friday/train.csv')

data.head()
print("Number of Rows: ", data.shape[0])
data.describe()
data.info()
print("Missing Values in Each Column:")

print(data.isna().sum())
fig, axes = plt.subplots(nrows=1, ncols=2)

sns.boxplot(data.Product_Category_2, ax=axes[0])

data.Product_Category_2.plot(kind='box', ax=axes[1])

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2)

sns.boxplot(data.Product_Category_3, ax=axes[0])

data.Product_Category_3.plot(kind='box', ax=axes[1])

plt.show()
data.fillna(method='ffill', inplace=True)

data.isna().sum()
data.fillna(method='bfill', inplace=True)

data.isna().sum()
print("Now lets see how's our data looking: ")

data.head(15)
data.info()
data.nunique()
# but first let confirm that is there any missing values left in a data

assert pd.notnull(data).all().all()
col = list(data.columns)

print("Unique Values in each column:\n")

for c in col:

    print(c, ": ", data[c].unique())

    print()
data['Gender'] = data.Gender.map({

    'M' : 0,

    'F' : 1

})

data.head()
data.Gender.unique()
from sklearn.preprocessing import LabelEncoder

lE = LabelEncoder()

data.City_Category = lE.fit_transform(data.City_Category)

data.head()
data.City_Category.unique()
data.info()
data.loc[data['Stay_In_Current_City_Years'] == '4+','Stay_In_Current_City_Years'] = '4'

data.Stay_In_Current_City_Years = data.Stay_In_Current_City_Years.astype('int64')

data.info()
print("Unique Values in Age Column:")

data.Age.unique()
sns.countplot(x=data.Age)

plt.show()
ax = sns.countplot(data.Gender)

gen = ['M', 'F']

ax.set(xticklabels=gen)

plt.show()
ax = sns.countplot(data.Marital_Status)

mar = ['Married', 'Single']

ax.set(xlabel='Martial Status', xticklabels=mar)

plt.show()
print('Martial Status: 0=Married, 1=Single')

print('Gender: 0=Male, 1=Female')

ax = data.groupby(['Marital_Status', 'Gender'])['Purchase'].count().plot(kind='bar')

ax.set(xlabel='Martial Status and Gender', ylabel='Purchase', title='Gender and Martial Status in respect to Purchase')

plt.show()
plt.bar(['PC1', 'PC2', 'PC3'], [data.Product_Category_1.sum(), data.Product_Category_2.sum(), data.Product_Category_3.sum()])

plt.show()
data.groupby('Stay_In_Current_City_Years')['Purchase'].count()
plt.bar([0, 1, 2, 3, 4], data.groupby('Stay_In_Current_City_Years')['Purchase'].count())

plt.show()
plt.bar(['A', 'B', 'C'], data.groupby('City_Category')['Purchase'].mean())

plt.show()
data.groupby('Gender')['Purchase'].mean()
sns.regplot(x='Gender', y='Purchase', data=data)

plt.show()
data.groupby('City_Category')['Purchase'].mean()
sns.regplot(x='City_Category', y='Purchase', data=data)

plt.show()
cols = ['Gender', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']

corr_result = data[cols].corr()

corr_result
sns.heatmap(corr_result, annot=True)

plt.show()
data.groupby(['Age', 'Marital_Status', 'Gender']).count()
print("Let see which user pays the maximum price for a product and for which Product:")

data.loc[data.Purchase.idxmax()]