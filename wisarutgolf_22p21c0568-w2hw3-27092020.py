# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from scipy.cluster import hierarchy

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
x = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

x = x[['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','price','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']]

x.fillna(0,inplace = True)

feat = []

for i in x.columns:

    feat.append(LabelEncoder().fit_transform(df[i])) if i == 'neighbourhood_group' or i == 'neighbourhood' or i == 'room_type' else feat.append(df[i])

    fear_arr = np.array(feat).T
def plot_dendrogram(n):

    link = hierarchy.linkage(fear_arr[:][:n],'complete')

    plt.figure(figsize = (25,15))

    dend = hierarchy.dendrogram(link)

    plt.show()
plot_dendrogram(22)