import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import re
dataset = pd.read_csv("../input/housedata/data.csv")
dataset.head(3)
dataset.describe(include="all")
dataset.info()
dataset.drop(columns=["date","country"], inplace=True)
dataset.head(3)
dataset.replace("NaN", np.nan, inplace = True)
missing_data = dataset.isnull()
missing_data.head(5)
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
dataset['city_num'] = lb_make.fit_transform(dataset["city"])
dataset[["city","city_num","price"]].head(3)
sns.regplot(x="city_num", y="price", data=dataset)
dataset[["city_num","price"]].corr()
dataset["statezip"].value_counts()
dataset['statezip_num'] = lb_make.fit_transform(dataset["statezip"])
sns.regplot(x="statezip_num", y="price", data=dataset)
dataset[["statezip_num","price"]].corr()
dataset.columns
corr=dataset[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
       'yr_built', 'yr_renovated']].corr()

ax = plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True,xticklabels=corr.columns.values,yticklabels=corr.columns.values)   
from scipy import stats
var = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
       'yr_built', 'yr_renovated']

pear_corr = []

for item in var:
  temp = stats.pearsonr(dataset[item], dataset['price'])[0]
  pear_corr.append(temp)
  
plt.figure(figsize=(20,5))
plt.title('pearson correlation between feature and price house')
ax = sns.barplot(x=var, y=pear_corr)
ax.set(xlabel='feature', ylabel='pearson correlation')

plt.show()
dataset[["yr_renovated","price"]].head(3)
dataset["yr_renovated"].value_counts().head(10)
dataset["yr_built"].value_counts()
plt.hist(dataset["yr_built"])

plt.xlabel("yr_built")
plt.ylabel("count")
plt.title("number of house based on year built")
bins = np.linspace(min(dataset["yr_built"]), max(dataset["yr_built"]), 5)
bins
group_names = ['early 90s', 'mid 90s', 'late 90s', '20s']
dataset['yr_built_binned'] = pd.cut(dataset['yr_built'], bins, labels=group_names, include_lowest=True )
dataset[['yr_built_binned','yr_built']].head(5)
dataset['yr_built_binned'].value_counts()
plt.hist(dataset["yr_built_binned"])

plt.xlabel("yr_built_binned")
plt.ylabel("count")
plt.title("number of house based on year built")
replace_year = {'temp': {'early 90s':1, 'mid 90s':2, 'late 90s':3, '20s':4}}
dataset["yr_built_binned"] = dataset["yr_built_binned"].astype('category') 
dataset["yr_built_binned_num2"] = dataset["yr_built_binned"].cat.codes
dataset["yr_built_binned_num2"].value_counts()
dataset['yr_built_binned_num2'] = lb_make.fit_transform(dataset["yr_built_binned"])
sns.regplot(x="yr_built_binned_num2", y="price", data=dataset)
dataset[["yr_built_binned_num2","price"]].corr()
dataset["condition"].value_counts()
sns.regplot(x="condition", y="price", data=dataset)
dataset[["condition", "price"]].corr()
var2 = ['price','bedrooms', 'bathrooms', 'sqft_living', 'floors',
       'waterfront', 'view', 'sqft_above', 'sqft_basement']
corr = dataset[var2].corr()
ax = plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True, xticklabels=corr.columns.values, yticklabels=corr.columns.values)   
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm
var3 = ['bedrooms', 'bathrooms', 'sqft_living', 'floors',
       'waterfront', 'view', 'sqft_above', 'sqft_basement']
lm.fit(dataset[var3], dataset['price'])
Y_hat = lm.predict(dataset[var3])
plt.figure(figsize=(10, 10))


ax1 = sns.distplot(dataset['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Predicted Values" , ax=ax1)


plt.title('Actual vs Fredicted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion')

plt.show()
plt.close()
lm.fit(dataset[var3], dataset['price'])
print('The R-square is: ', lm.score(dataset[var3], dataset['price']))