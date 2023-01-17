# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
file_path='../input/google-play-store-apps/googleplaystore.csv'
data = pd.read_csv(file_path)
data.drop_duplicates(subset='App', inplace=True)
#группировка по категории
datas=data.groupby(['Category'])['Rating','Category'].mean()
datas.sort_values(by=['Rating'], ascending=False,inplace=True)
datas.reset_index(level=0, inplace=True)
datas.head()
#берем выборку по лучшим 10 категориям
view = datas.iloc[1:11]
view.head()
plt.figure(figsize=(35,6))

#график по категории и рейтингу 
sns.barplot(x=view['Category'], y=view['Rating'])
plt.ylabel("рейтинг")
plt.xlabel("категории")