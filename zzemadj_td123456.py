# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

d1=pd.read_csv("../input/kc_house_data.csv",",")

d1.dtypes

#print(d1.describe(include='all'))

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
d1['bedrooms']=d1['bedrooms'].astype('category')

d1['floors']=d1['floors'].astype('category')

d1['price']=d1['price'].astype('int64')

d1['yr_cat']=pd.cut(d1['yr_built'],bins=5,labels=["old old","old","med","new","new new"])

tab=d1.pivot_table (['price'],['bedrooms','yr_cat'],aggfunc='sum')/d1.pivot_table (['price'],['bedrooms','yr_cat'],aggfunc='count')

tab2=tab.unstack()

tab2

tab2.plot()

#tab2=tab.astype('int64')

#plt.figure(1)

#plt.plot(tab['yr_cat'], 'k')

#plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

#plt.show()
sns.set()

count_year=d1.pivot_table (['price'],['yr_cat'],aggfunc='count')

#count_year.hist()

#g = sns.FacetGrid(count_year)  

#g.map(sns.distplot,count_year['price'].values.tolist())  

#sns.distplot(count_year['price'],kde=False)

#count_year

#count_year['price'].plot()