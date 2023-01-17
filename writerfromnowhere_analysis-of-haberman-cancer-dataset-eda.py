# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
haberman = pd.read_csv("/kaggle/input/habermans-survival-data-set/haberman.csv")
#haberman.columns
haberman.columns = ['age','year','nodes','status']
haberman.columns
haberman.status.value_counts()
# mean age 
print("Mean of age = {0:.1f}".format(haberman.age.mean()))
# median age
print("Median of age = {0:.1f}".format(haberman.age.median()))
print("Max age = {0:.1f}".format(haberman.age.max()))
print("-"*50)
# mean age 
print("Mean of nodes = {0:.1f}".format(haberman.nodes.mean()))
# median age
print("Median of nodes= {0:.1f}".format(haberman.nodes.median()))
print("Max nodes = {0:.1f}".format(haberman.nodes.max()))
print("-"*50)
print("Mean of years = {0:.1f}".format(haberman.year.mean()))
# median age
print("Median of years= {0:.1f}".format(haberman.year.median()))
print("Max years = {0:.1f}".format(haberman.year.max()))
print("-"*50)

haberman.plot(kind='scatter',x='age',y='year')
plt.show()
sns.FacetGrid(haberman,hue='status',height=7)\
    .map(plt.scatter,'age','year')\
    .add_legend()
plt.show()
sns.set_style('whitegrid')
sns.pairplot(haberman,hue='status',height=5)
plt.show()
sns.FacetGrid(haberman,hue='status',height=7)\
    .map(plt.scatter,'age','nodes')\
    .add_legend()
plt.show()
sns.FacetGrid(haberman,hue='status',height=7)\
    .map(plt.scatter,'nodes','year')\
    .add_legend()
plt.show()
sns.FacetGrid(haberman,hue='status',height=7)\
    .map(sns.distplot,"age")\
    .add_legend()
plt.show()
sns.FacetGrid(haberman,hue='status',height=7)\
    .map(sns.distplot,"year")\
    .add_legend()
plt.show()
sns.FacetGrid(haberman,hue='status',height=7)\
    .map(sns.distplot,"nodes")\
    .add_legend()
plt.show()

counts , bin_ages = np.histogram(haberman[haberman.status ==1].nodes,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)



plt.show()
counts , bin_ages = np.histogram(haberman[haberman.status ==2].nodes,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)



plt.show()

counts , bin_ages = np.histogram(haberman[haberman.status == 1].nodes,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)

counts , bin_ages = np.histogram(haberman[haberman.status == 2].nodes,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)


plt.show()

counts , bin_ages = np.histogram(haberman[haberman.status == 1].age,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)

counts , bin_ages = np.histogram(haberman[haberman.status == 2].age,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)


plt.show()
print("Standard deviation of nodes for people who survived  ")
print(np.std(haberman[haberman.status==1].nodes))

print("Standard deviation of nodes for people who didn't survive ")
print(np.std(haberman[haberman.status==2].nodes))
print("-"*50)

print("Standard deviation of ages for people who survived")
print(np.std(haberman[haberman.status==1].age))

print("Standard deviation of ages for people who didn't survive")
print(np.std(haberman[haberman.status==2].age))
print("-"*50)

from statsmodels import robust
print("Median absolute deviation of nodes for people who survived")
print(robust.mad(haberman[haberman.status==1].nodes))

print("Median absolute deviation of nodes for people who didin't survived")
print(robust.mad(haberman[haberman.status==2].nodes))
print("-"*50)

print("Median absolute deviation of age for people who survived ")
print(robust.mad(haberman[haberman.status==1].age))

print("Median absolute deviation of age for people who didn't survived ")
print(robust.mad(haberman[haberman.status==2].age))
print("-"*50)
print("Median of nodes for people who survived ")
print(np.median(haberman[haberman.status==1].nodes))

print("Median of nodes of people who didn't survive ")
print(np.median(haberman[haberman.status==2].nodes))
print("-"*50)

print("Median of age of people who survived")
print(np.median(haberman[haberman.status==1].age))

print("Median of age of people who didn't survive ")
print(np.median(haberman[haberman.status==2].age))
print("-"*50)

print("Quantiles of nodes for people who survived")
print(np.percentile(haberman[haberman.status==1].nodes,np.arange(0,100,25)))
print("90th percentile of nodes for people who survived")
print(np.percentile(haberman[haberman.status==1].nodes,90))
print("\n")
print("Quantiles of nodes for people who didn't survive")
print(np.percentile(haberman[haberman.status==2].nodes,np.arange(0,100,25)))
print("90th percentile of nodes for people who didn't survive")
print(np.percentile(haberman[haberman.status==2].nodes,90))
print("-"*50)
print("Quantiles of age for people who survived")
print(np.percentile(haberman[haberman.status==1].age,np.arange(0,100,25)))
print("90th percentile of age for people who survived")
print(np.percentile(haberman[haberman.status==1].age,90))
print("\n")
print("Quantiles of age for people who didn't survive")
print(np.percentile(haberman[haberman.status==2].age,np.arange(0,100,25)))
print("90th percentile of age for people who didn't survive")
print(np.percentile(haberman[haberman.status==2].age,90))
sns.boxplot(x='status',y='nodes',data=haberman)
plt.show()
sns.boxplot(x='status',y='age',data=haberman)
plt.show()
sns.violinplot(x='status',y='nodes',data=haberman)
plt.show()
sns.violinplot(x='status',y='age',data=haberman)
plt.show()
sns.jointplot(x='age',y='nodes',data=haberman[haberman.status==1],kind='kde')
plt.show()
sns.jointplot(x='age',y='nodes',data=haberman[haberman.status==2],kind='kde')
plt.show()