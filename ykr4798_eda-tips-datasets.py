# importing important lib.



import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
# Reading a dataset



df = pd.read_excel("../input/eda-dataset/Tips Quick EDA exercise v0.1 (5).xlsx")



# Return first n rows(default 5 rows) of dataset.



df.head()
# Return last n rows(default 5 rows) of dataset.



df.tail()
# The describe() method is used for calculating some statistical data like percentile, mean and std of the numerical values of 

# the Series or DataFrame.



df.describe()
# Print a concise summary of a DataFrame.



df.info()
Avg = df['tip'].mean()

Avg                                     # Avg = 3 (approx)
Des = df['tip'].describe()

Med = df['tip'].median()

print(Des, "\nMedian=",Med)
# for ignoring warning



import warnings;

warnings.simplefilter('ignore') # supress warning



# C:\Users\ykr47\anaconda3\lib\site-packages\matplotlib\cbook\__init__.py:1239: RuntimeWarning: invalid value encountered in less_equal

#   wiskhi = x[x <= hival]

# C:\Users\ykr47\anaconda3\lib\site-packages\matplotlib\cbook\__init__.py:1246: RuntimeWarning: invalid value encountered in greater_equal

#   wisklo = x[x >= loval]

# C:\Users\ykr47\anaconda3\lib\site-packages\matplotlib\cbook\__init__.py:1254: RuntimeWarning: invalid value encountered in less

#   x[x < stats['whislo']],

# C:\Users\ykr47\anaconda3\lib\site-packages\matplotlib\cbook\__init__.py:1255: RuntimeWarning: invalid value encountered in greater

#   x[x > stats['whishi']],



Graph_tip = df['tip']                      

sns.boxplot(Graph_tip)

plt.title('Box plot')

plt.show()



# another way 

# df['tip'].plot.box()

# plt.show()
df[df.tip>=7]
Graph_total_bill = df['total_bill']                      

sns.boxplot(Graph_total_bill)

plt.title('Box plot')

plt.show()



# another way 

# df['total_bill'].plot.box()

# plt.show()
df[df.total_bill>=40]
Percent_Female = df['sex'].value_counts(normalize = True)

Percent_Female
df["sex"].value_counts(dropna=False)
# To calculate this in pandas with the value_counts() method, set the argument normalize to True.



df['sex'].value_counts(normalize=True)
Percentage = df['sex'].value_counts(normalize=True)

Percentage
# this graph shows the count of male/female.



df['sex'].value_counts(normalize=True).plot.bar()

plt.title('Ratio of Male/Female')

plt.xlabel("sex")

plt.ylabel("Percent")

plt.xticks(rotation=0)

plt.show()
df.groupby(['sex'])['tip'].mean().plot.bar()

plt.title('tip differ by gender')

plt.ylabel("tip")

plt.xticks(rotation=0)

plt.show()
df.groupby(['time'])['tip'].mean().plot.bar()

plt.title('tip differ by the time of day')

plt.ylabel("Average tip count")

plt.xticks(rotation=0)

plt.show()
df.groupby(['size'])['tip'].mean().plot.bar()

plt.title('tip differ by size')

plt.ylabel("Average tip count")

plt.xticks(rotation=0)

plt.show()
df.groupby(['smoker'])['tip'].count().plot.bar()

plt.title('tip with respect to smoking ')

plt.ylabel("tip count")

plt.xticks(rotation=0)

plt.show()
df.groupby(['smoker','sex'])['tip'].mean().plot.bar()

plt.title('Gender vs. smoker/non-smoker and tip size ')

plt.ylabel("Average tip count")

plt.xticks(rotation=0)

plt.show()
df.groupby(['smoker','sex'])['tip'].mean().unstack()   # unstack() make a distorted data to organised data.
df.insert(2,"pct_tip",df.tip/df.total_bill)
# Return first n rows(default 5 rows) of dataset.



df.head()
df.groupby(['sex'])['pct_tip'].count().plot.bar()

plt.title('pct_tip differ by gender')

plt.ylabel("Average tip count")

plt.xticks(rotation=0)

plt.show()
df.groupby(['sex','smoker'])['pct_tip'].mean().unstack()   # unstack() make a distorted data to organised data.
df.groupby(['size'])['pct_tip'].count().plot.bar()

plt.title('pct_tip differ by size')

plt.ylabel("Average tip count")

plt.xticks(rotation=0)

plt.show()
df.groupby(['smoker','sex'])['pct_tip'].mean().plot.bar()

plt.title('gender vs. smoker view using pct_tip ')

plt.ylabel("Average pct_tip count")

plt.xticks(rotation=0)

plt.show()
df.plot.scatter(x='total_bill', y='tip')

plt.xlabel('Total Bill')            

plt.ylabel('Tip')

plt.title('Total Bill vs Tip')            

plt.show()
df.plot.scatter(x='total_bill', y='pct_tip')

plt.xlabel('Total Bill')              # label = name of label

plt.ylabel('Percentage Tip')

plt.title('Total Bill vs Percentage Tip Scatter Plot')            # title = title of plot

plt.show()