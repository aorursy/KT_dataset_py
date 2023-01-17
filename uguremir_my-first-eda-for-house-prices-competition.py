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

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Import train.csv into data

data = pd.read_csv("../input/train.csv")
#Check whether data is uploaded 

data.head()
# 1. Print name of columns

column_names = data.columns

column_names
# 1.2 Print types of features

data.info()
# 2. Number of feature that we have in our dataset

column_names.value_counts().sum() - 1 # Minus 1, because Id column is not considered as feature.
# 3. Our target value is "SalePrice", Let's visualize it

sns.set(font_scale=1)

plt.figure(figsize=(10,8))

sns.distplot(data["SalePrice"],hist=True,kde=True,bins=int(100/1.5),kde_kws={'linewidth': 4},color = 'darkblue',hist_kws={'edgecolor':'black'})

plt.xticks(rotation=45)

plt.xticks(np.arange(0, 850000, step=50000))

plt.show()

# 4.1 SalePrice relationship with other numerical variables.

var = "LotArea"

vis_data = pd.concat([data['SalePrice'],data[var]],axis=1)

plt.subplots(figsize=(8,6))

sns.scatterplot(x=var,y='SalePrice',data=vis_data);



# 3. Our target value is "SalePrice", Let's visualize it

sns.set(font_scale=1)

plt.figure(figsize=(8,6))

sns.distplot(data[data[var]>0][var],hist=True,kde=True,bins=int(100/1.5),kde_kws={'linewidth': 4},color = 'darkblue',hist_kws={'edgecolor':'black'})

plt.xticks(rotation=45)

plt.show()
# 4.2 SalePrice relationship with other numerical variables.

var = "LotFrontage"

vis_data = pd.concat([data['SalePrice'],data[var]],axis=1)

plt.subplots(figsize=(8,6))

sns.scatterplot(x=var,y='SalePrice',data=vis_data);



# 3. Our target value is "SalePrice", Let's visualize it

sns.set(font_scale=1)

plt.figure(figsize=(8,6))

sns.distplot(data[data[var]>0][var],hist=True,kde=True,bins=int(100/1.5),kde_kws={'linewidth': 4},color = 'darkblue',hist_kws={'edgecolor':'black'})

plt.xticks(rotation=45)

plt.show()



# 4.3.1 SalePrice relationship with other numerical variables.

var = "MasVnrArea"

vis_data = pd.concat([data['SalePrice'],data[var]],axis=1)

plt.subplots(figsize=(8,6))

sns.scatterplot(x=var,y='SalePrice',data=vis_data);

# 4.3.2 SalePrice relationship with other numerical variables.

var = "MasVnrArea"

vis_data = pd.concat([data['SalePrice'],data[data[var]>0][var]],axis=1)

plt.subplots(figsize=(8,6))

sns.scatterplot(x=var,y='SalePrice',data=vis_data);



# 3. Our target value is "SalePrice", Let's visualize it

sns.set(font_scale=1)

plt.figure(figsize=(8,6))

sns.distplot(data[data[var]>0][var],hist=True,kde=True,bins=int(100/1.5),kde_kws={'linewidth': 4},color = 'darkblue',hist_kws={'edgecolor':'black'})

plt.xticks(rotation=45)

plt.show()

# 4.4 SalePrice relationship with other numerical variables.

var = "GrLivArea"

vis_data = pd.concat([data['SalePrice'],data[var]],axis=1)

plt.subplots(figsize=(8,6))

sns.scatterplot(x=var,y='SalePrice',data=vis_data);



# 3. Our target value is "SalePrice", Let's visualize it

sns.set(font_scale=1)

plt.figure(figsize=(8,6))

sns.distplot(data[var],hist=True,kde=True,bins=int(100/1.5),kde_kws={'linewidth': 4},color = 'darkblue',hist_kws={'edgecolor':'black'})

plt.xticks(rotation=45)

plt.show()


# 4.5 SalePrice relationship with other numerical variables.

var = "GarageArea"

vis_data = pd.concat([data['SalePrice'],data[var]],axis=1)

plt.subplots(figsize=(8,6))

sns.scatterplot(x=var,y='SalePrice',data=vis_data);



# 3. Our target value is "SalePrice", Let's visualize it

sns.set(font_scale=1)

plt.figure(figsize=(8,6))

sns.distplot(data[var],hist=True,kde=True,bins=int(100/1.5),kde_kws={'linewidth': 4},color = 'darkblue',hist_kws={'edgecolor':'black'})

plt.xticks(rotation=45)

plt.show()
var = "Street"

vis_data = pd.concat([data['SalePrice'],data[var]],axis=1)

plt.subplots(figsize=(8,6))

sns.boxplot(x=var,y='SalePrice',data=vis_data);
var = "LotShape"

vis_data = pd.concat([data['SalePrice'],data[var]],axis=1)

plt.subplots(figsize=(8,6))

sns.boxplot(x=var,y='SalePrice',data=vis_data);


var = "HouseStyle"

vis_data = pd.concat([data['SalePrice'],data[var]],axis=1)

plt.subplots(figsize=(8,6))

sns.boxplot(x=var,y='SalePrice',data=vis_data);


var = "RoofStyle"

vis_data = pd.concat([data['SalePrice'],data[var]],axis=1)

plt.subplots(figsize=(8,6))

sns.boxplot(x=var,y='SalePrice',data=vis_data);



var = "RoofStyle"

vis_data = pd.concat([data['SalePrice'],data[var]],axis=1)

plt.subplots(figsize=(8,6))

sns.violinplot(x=var,y='SalePrice',data=vis_data);


var = "SaleType"

vis_data = pd.concat([data['SalePrice'],data[var]],axis=1)

plt.subplots(figsize=(8,6))

sns.boxplot(x=var,y='SalePrice',data=vis_data);

# 5 Correlation Matrix



cormap = data.corr()

sns.set(font_scale=1)

plt.subplots(figsize=(15,13))

sns.heatmap(cormap,linewidths=1,square=True)
# 6. 10 highest correlated features with SalePrice

cols = cormap.nlargest(10,'SalePrice')['SalePrice'].index

cormap10 = data[cols].corr()

plt.subplots(figsize=(10,8))

sns.heatmap(cormap10,linewidths=1,square=True, annot=True)
# 6. Pairplot



sns.pairplot(data[cols],size=2)
# 7.1 Missing Data



total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

col_types = data[total.index].dtypes

missing_data = pd.concat([total,percent,col_types],axis=1)

missing_data.columns=['Total','Percent','Col_Types']

missing_data.head(20)

# 7.2 Missing Data



missing_data.head(19)[missing_data.head(19)['Col_Types'] == 'float64']
# 7.3 Missing Data



missing_data.head(19)[missing_data.head(19)['Col_Types'] == 'object']
#7.4 Remove missing data



remove_col_names = total.head(18).index

new_data = data.drop(remove_col_names,axis=1)

new_data = new_data.dropna()

#7.4 Check there is still null values or not



total = new_data.isnull().sum().sort_values(ascending=False)

count = new_data.isnull().count()

clean_data = pd.concat([total,count],axis=1)

clean_data.columns = ['Count','Total']

clean_data.sort_values(by='Count',ascending=False)
#8.1 Transformation

log_data = new_data
#8.1 Transformation

from scipy import stats



var = "SalePrice"



plt.figure(figsize=(12,8))

plt.suptitle("Before Transformation")

plt.subplot(1,2,1)

sns.distplot(log_data[var])

plt.subplot(1,2,2)

res = stats.probplot(log_data[var], plot=plt)





log_data[var] = np.log10(log_data[var])



plt.figure(figsize=(12,8))

plt.suptitle("After Transformation")

plt.subplot(1,2,1)

sns.distplot(log_data[var])

plt.subplot(1,2,2)

res = stats.probplot(log_data[var], plot=plt)
var = "LotArea"



plt.figure(figsize=(12,8))

plt.suptitle("Before Transformation")

plt.subplot(1,2,1)

sns.distplot(log_data[var])

plt.subplot(1,2,2)

res = stats.probplot(log_data[var], plot=plt)





log_data[var] = np.log10(log_data[var])



plt.figure(figsize=(12,8))

plt.suptitle("After Transformation")

plt.subplot(1,2,1)

sns.distplot(log_data[var])

plt.subplot(1,2,2)

res = stats.probplot(log_data[var], plot=plt)
var = "GrLivArea"



plt.figure(figsize=(12,8))

plt.suptitle("Before Transformation")

plt.subplot(1,2,1)

sns.distplot(log_data[var])

plt.subplot(1,2,2)

res = stats.probplot(log_data[var], plot=plt)





log_data[var] = np.log10(log_data[var])



plt.figure(figsize=(12,8))

plt.suptitle("After Transformation")

plt.subplot(1,2,1)

sns.distplot(log_data[var])

plt.subplot(1,2,2)

res = stats.probplot(log_data[var], plot=plt)
var = "GarageArea"



plt.figure(figsize=(12,8))

plt.suptitle("Before Transformation")

plt.subplot(1,2,1)

sns.distplot(log_data[var])

plt.subplot(1,2,2)

res = stats.probplot(log_data[var], plot=plt)





log_data[var] = np.sqrt(log_data[var])



plt.figure(figsize=(12,8))

plt.suptitle("After Transformation")

plt.subplot(1,2,1)

sns.distplot(log_data[var])

plt.subplot(1,2,2)

res = stats.probplot(log_data[var], plot=plt)
#9 Dummy Variables

pd.get_dummies(log_data)