import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/toy-dataset/toy_dataset.csv")

data.head()
data.shape
data.isnull().sum()

#No null values in the dataset
data.dtypes
#Make 'Number' column the index of the dataframe

data.set_index('Number',inplace = True)
data.head()
data.City.value_counts()
data.Gender.value_counts()
data.Illness.value_counts()
data.City.value_counts().plot.bar()

plt.show()
data.Gender.value_counts().plot.bar()

plt.show()
data.Illness.value_counts().plot.bar()

plt.show()
sns.distplot(data.Age,hist_kws=dict(edgecolor="k", linewidth=1,color='grey'),color='red')

plt.show()

#All the age groups are equally represented in the dataset
sns.distplot(data.Income,hist_kws=dict(edgecolor="k", linewidth=1,color='grey'),color='red')

plt.show()
data.groupby(['City']).mean().Income.plot.bar()

plt.show()

#Average income is maximum for Mountain View.

#This can be attributed to it being one of the cities of Silicon Valley
data.groupby(['City']).mean().Age.plot.bar()

plt.show()

#Average age of population is almost similar for all the cities
data['Count'] = data.Illness.apply(lambda x : 1 if x == 'Yes' else 0)

data.Count.value_counts()
data.groupby(['City']).mean().Count.plot.barh()

plt.show()

#Average health for all the cities is nearly same
plt.scatter(x = data.Age,y = data.Income)

plt.show()
data.groupby(['Gender']).mean().Income.plot.bar()

plt.show()
pvt_tbl = pd.pivot_table(data = data,index = 'Gender',columns = 'City', values = 'Income', aggfunc = np.mean)

plt.figure(figsize = [10,6])

sns.heatmap(pvt_tbl,cmap = 'Greens',annot = True)

plt.show()
#Normalized values for heatmap

crs_tb = pd.crosstab(data.Gender,data.City, values = data.Income, aggfunc = np.mean,normalize = True)

plt.figure(figsize = [10,6])

sns.heatmap(crs_tb,cmap = 'Paired',annot = True)

plt.show()