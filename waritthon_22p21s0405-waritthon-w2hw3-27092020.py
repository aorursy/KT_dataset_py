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
%matplotlib inline
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
Data= pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
Data.head(5)
Data.info()
n_row, n_col = Data.shape

print("Number of row: ",n_row)
print("Number of column: ",n_col)
fig = plt.subplots(figsize = (12,5))
sns.countplot(x = 'room_type', hue = 'neighbourhood_group', data = Data)
Data['neighbourhood_group'].value_counts()
Data.describe()
plt.figure(figsize = (16,12))
Data_set = Data[Data.price < Data.price.quantile(0.99)]
sns.violinplot(x = 'neighbourhood_group', y = 'price', data = Data_set)
Data_Gprice = Data.groupby('neighbourhood').mean().reset_index().loc[:,['neighbourhood','price']]
list_Gprice_mean = [ [price] for price in Data_Gprice.price.to_list()]
Data_Gprice.head(10)
fig = plt.figure(figsize=(60, 50))
dendrogram(linkage(list_Gprice_mean, 'complete'), labels=Data_Gprice.neighbourhood.to_list(), leaf_rotation=0, orientation="right")
plt.show()
