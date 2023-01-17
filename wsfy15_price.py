# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

click_price = pd.read_csv("../input/price.txt", header=None, names=['price'], low_memory=False) #, dtype={'price':float}
click_price.head()

click_price['price'].value_counts()

price = pd.read_csv("../input/price3.txt", header=None, names=['price'], low_memory=False) #, dtype={'price':float}
price['price'].value_counts()
# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
#df['price'] = float(df['price'])
#df['pirce'].dtype
#sns.histplot()
#price.plot().line()
price.describe()
price['price'].max()
fig1, axarr1 = plt.subplots(nrows=2, ncols=1, figsize=(100, 20))
price['price'].value_counts().sort_index().plot.bar(
    ax=axarr1[0]
)
axarr1[0].set_title('all price')
click_price['price'].value_counts().sort_index().plot.bar(
    ax=axarr1[1]
)     
axarr1[1].set_title('clicked price')
#ax.xaxis.tick_top()
#ax.tick_params(labelsize=8)
#ax.tick_params(axis='x',labelsize=20, colors='b', labeltop=True, labelbottom=False)
fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(100, 20))
price['price'].value_counts().head(200).sort_index().plot.bar(
    ax=axarr[0]
)
axarr[0].set_title('The top 200 in all price')
click_price['price'].value_counts().head(200).sort_index().plot.bar(
    ax=axarr[1]
)     
axarr[1].set_title('The top 200 in clicked price')
# plt.show()