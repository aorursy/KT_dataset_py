# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/habermans-survival-data-set/haberman.csv')
df.head()
df.columns = ['Age','Op_year','Axil_nodes','Survival_status']
df.head()
df.Survival_status = df.Survival_status.replace({1:'Survived',2:'Died'})
df.head()
df.tail()
df.Survival_status.value_counts()
df.shape
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.pairplot(df,hue='Survival_status')
plt.show()
sns.FacetGrid(df,hue="Survival_status")\
    .map(sns.distplot,'Age').add_legend();
sns.FacetGrid(df,hue='Survival_status').map(sns.distplot,'Axil_nodes').add_legend()
sns.FacetGrid(df,hue='Survival_status').map(sns.distplot,'Op_year')
sns.FacetGrid(df,hue='Survival_status',height=5).map(plt.scatter,'Age','Axil_nodes').add_legend()
df.shape
df.Survival_status.value_counts()
died_df = df.loc[df['Survival_status'] == 'Died']
died_df.shape
shuffled_df = df.sample(frac = 1,random_state=4)
shuffled_df.shape
died_df = shuffled_df.loc[shuffled_df['Survival_status'] ==  'Died']
died_df.shape
survived_df = shuffled_df.loc[shuffled_df['Survival_status'] == 'Survived'].sample(n=81)

survived_df.shape
normalized_df = pd.concat([died_df,survived_df])
normalized_df.shape
sns.FacetGrid(normalized_df,height=6,hue='Survival_status').map(plt.scatter,'Axil_nodes','Age').add_legend()
sns.FacetGrid(normalized_df,hue='Survival_status',height=6).map(sns.distplot,'Axil_nodes').add_legend()
sns.FacetGrid(normalized_df,hue='Survival_status',height=6).map(sns.distplot,'Age').add_legend()
sns.FacetGrid(normalized_df,hue='Survival_status',height=6).map(sns.distplot,'Op_year').add_legend()
sns.pairplot(normalized_df,hue='Survival_status',height=8,markers=["o", "s"])

# For Age
import numpy as np
counts, bin_edges = np.histogram(survived_df.Age,bins=10,density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.subplot(2,2,1)
plt.plot(bin_edges[1:],pdf,label = 'Age-PDF-Survived')
plt.legend()
plt.subplot(2,2,2)
plt.plot(bin_edges[1:],cdf,label = 'Age-CDF-Survived')
plt.legend()
counts, bin_edges = np.histogram(died_df.Age,bins=10,density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.subplot(2,2,3)
plt.plot(bin_edges[1:],pdf,label = 'Age-PDF-Died')
plt.legend()
plt.subplot(2,2,4)
plt.plot(bin_edges[1:],cdf,label = 'Age-CDF-Died')
plt.legend()
plt.show()
counts, bin_edges = np.histogram(survived_df.Axil_nodes,bins=10,density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.subplot(2,2,1)
plt.plot(bin_edges[1:],pdf,label = 'Axil-PDF-Survived')
plt.legend()
plt.subplot(2,2,2)
plt.plot(bin_edges[1:],cdf,label = 'Axil-CDF-Survived')
plt.legend()
counts, bin_edges = np.histogram(died_df.Axil_nodes,bins=10,density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.subplot(2,2,3)
plt.plot(bin_edges[1:],pdf,label = 'Axil-PDF-Died')
plt.legend()
plt.subplot(2,2,4)
plt.plot(bin_edges[1:],cdf,label = 'Axil-CDF-Died')
plt.legend()
plt.show()
sns.boxplot(data=normalized_df,x='Survival_status',y='Age')
sns.boxplot(data = normalized_df,x='Survival_status',y='Op_year')
sns.boxplot(data = normalized_df, x = 'Survival_status', y = 'Axil_nodes')
sns.violinplot(data = normalized_df, x = 'Survival_status', y = 'Axil_nodes')
sns.jointplot(data = survived_df, x = 'Age', y = 'Axil_nodes')
sns.jointplot(data = survived_df, x = 'Age', y = 'Axil_nodes',kind = 'hex')
sns.jointplot(data = survived_df, x = 'Age', y = 'Axil_nodes',kind='kde')
sns.jointplot(data = died_df, x = 'Age', y = 'Axil_nodes',kind='kde')
