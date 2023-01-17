# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/ks-projects-201801.csv")
df = df[['category', 'main_category', 'deadline',
       'launched', 'country', 'usd_pledged_real', 
       'usd_goal_real', 'state', 'backers']]
df.head()
df[['main_category', 'state']].pivot_table(columns='state', index='main_category', aggfunc=np.size).plot(kind='bar', figsize=(8, 8), stacked=True)
df1 = df[['main_category', 'usd_pledged_real', 'usd_goal_real', 'backers', 'state']]
df1 = pd.concat([df1, pd.get_dummies(df1['state'])], axis=1).drop('state', axis=1)
df1 = df1.groupby('main_category').aggregate([np.sum, np.mean])
df1
df1.apply(lambda x: (x / (np.sum(x)))).plot.pie(subplots=True, figsize=(15*14,10))
sns.heatmap(df1.apply(lambda x: (x / (np.sum(x)))))