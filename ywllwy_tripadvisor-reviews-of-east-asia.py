# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import datetime



import matplotlib.pyplot as plt

import matplotlib.patches as mpatches



import seaborn as sns



%matplotlib inline



review = pd.read_csv("../input/tripadvisor-reviews/tripadvisor_review.csv")

review.head()

review.shape
review.info()
RV = review.rename({"Category 1":"Art Galleries", "Category 2":"Dance Clubs", "Category 3":"Juice Bars", "Category 4":"Restaurants", "Category 5":"Museums", "Category 6":"Resorts", "Category 7":"Parks/Picnic Spots", "Category 8":"Beaches", "Category 9":"Theaters", "Category 10":"Religious Institutions"}, axis=1)
RV
b = RV.describe()

b
plt.figure(figsize=(50,25)).suptitle('Category Rating Average', fontsize=80)



def pltcolor(lst):

    cols=[]

    for x in lst:

        if x <= 1.0:

            cols.append('r')

        elif x > 1.0 and x <= 2.0:

            cols.append('m')

        elif x > 2.0 and x <= 3.0:

            cols.append('c')

        else:

            cols.append('g')

    return cols

crl=pltcolor(b.iloc[1,:])



sns.barplot(x=list(RV)[1:], y=b.iloc[1,:], palette=crl)



plt.xlabel('Category', fontsize=50)

plt.ylabel('Avg. Rating', fontsize=50)

plt.xticks(fontsize=30, rotation=0)

plt.yticks(fontsize=40, rotation=0)



r_patch = mpatches.Patch(color='r', label='0.0 - 1.0')

m_patch = mpatches.Patch(color='m', label='1.0 - 2.0')

c_patch = mpatches.Patch(color='c', label='2.0 - 3.0')

g_patch = mpatches.Patch(color='g', label='3.0 - 4.0')



plt.legend(handles=[r_patch, m_patch, c_patch, g_patch], loc="upper left", fontsize=50)



plt.show()
fig = plt.figure(figsize = (15,15))

ax = fig.gca()

RV.hist(ax=ax, bins=[0, 1, 2, 3, 4])



plt.show()
plt.subplots(figsize=(10, 8))

sns.heatmap(RV.corr(), vmax=.8, square=True);


plt.subplots(figsize=(20, 15))

k = 10 

cols = RV.corr().nlargest(k, 'Art Galleries')['Art Galleries'].index

cm = np.corrcoef(RV[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 20}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()