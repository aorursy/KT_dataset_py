import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import matplotlib
from matplotlib import cm

matplotlib.rcParams['figure.figsize'] = (20, 10)
print(os.listdir("../input"))
us = pd.read_csv('../input/USvideos.csv')
# Took help from quannguyen135 kernel. Thanks 
us['trending_date'] = pd.to_datetime(us['trending_date'], format='%y.%d.%m')
us['publish_time'] = pd.to_datetime(us['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
us.insert(4, 'publish_date', us['publish_time'].dt.date)
us['publish_time'] = us['publish_time'].dt.time
us.head()
us['t_month'] = pd.DatetimeIndex(us['trending_date']).month
us['t_year'] = pd.DatetimeIndex(us['trending_date']).year
comments_year_wise = us.groupby('t_year')['comment_count'].sum()

sns.barplot(comments_year_wise.index, comments_year_wise.values)
plt.xlabel("Year")
plt.ylabel("Total Comments")
plt.show()
likes_year_wise = us.groupby('t_year')['likes'].sum()
sns.barplot(likes_year_wise.index, likes_year_wise.values)
plt.xlabel("Year")
plt.ylabel("Total Likes")
plt.show()









