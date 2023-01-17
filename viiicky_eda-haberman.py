# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
haberman = pd.read_csv('../input/haberman.csv')
print('Number of points, number of attributes: {}'.format(haberman.shape))
print('Features and classes: {}'.format(haberman.columns))
print('30 : Age')
print('64 : Op_Year')
print('1 : axil_nodes_det')
print('1.1 : Surv_status (Class attribute) 1 = the patient survived 5 years or longer; 2 = the patient died within 5 year')
print('Domain Knowledge:')
print('Positive axillary lymph node: A positive axillary lymph node is a lymph node in the area of the armpit (axilla) to which cancer has spread. This spread is determined by surgically removing some of the lymph nodes and examining them under a microscope to see whether cancer cells are present.')
print('Count of survived(1)/unsurvived(2) patients:')
count =  haberman['1.1'].value_counts()
print(count)
sns.FacetGrid(haberman, hue='1.1', height=5).map(sns.distplot, '30').add_legend()
plt.show()
sns.FacetGrid(haberman, hue='1.1', height=5).map(sns.distplot, '64').add_legend()
plt.show()
sns.FacetGrid(haberman, hue='1.1', height=5).map(sns.distplot, '1').add_legend()
plt.show()
hist, bin_edges = np.histogram(haberman.loc[haberman['1.1']==1, '30'], density=True)
pdf = hist/sum(hist)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

hist, bin_edges = np.histogram(haberman.loc[haberman['1.1']==2, '30'], density=True)
pdf = hist/sum(hist)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

plt.grid(True)
plt.show()
hist, bin_edges = np.histogram(haberman.loc[haberman['1.1']==1, '64'], density=True)
pdf = hist/sum(hist)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

hist, bin_edges = np.histogram(haberman.loc[haberman['1.1']==2, '64'], density=True)
pdf = hist/sum(hist)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

plt.grid(True)
plt.show()
hist, bin_edges = np.histogram(haberman.loc[haberman['1.1']==1, '1'], density=True)
pdf = hist/sum(hist)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

hist, bin_edges = np.histogram(haberman.loc[haberman['1.1']==2, '1'], density=True)
pdf = hist/sum(hist)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

plt.grid()
plt.show()
sns.boxplot(x='1.1', y='30', data=haberman)
plt.show()
sns.boxplot(x='1.1', y='64', data=haberman)
plt.show()
sns.boxplot(x='1.1', y='1', data=haberman)
plt.show()
sns.violinplot(x='1.1', y='30', data=haberman)
plt.show()
sns.violinplot(x='1.1', y='64', data=haberman)
plt.show()
sns.violinplot(x='1.1', y='1', data=haberman)
plt.show()
sns.pairplot(haberman, hue='1.1', vars=['30', '64', '1'], size=3)
plt.show()