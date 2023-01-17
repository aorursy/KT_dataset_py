# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/top-spotify-songs-from-20102019-by-year/top10s.csv", encoding='ISO-8859-1', index_col=0)
data.head()
data.isnull().sum()
df1 = data.groupby(['year']).median()
df1['bpm']
data.groupby(['year']).max()[['artist','title']]
import seaborn as sns
sns.lineplot(x='year', y='bpm', data = data)
sns.lineplot(x='year', y='dB', data = data)
sns.lineplot(x='year', y='dur', data = data)
sns.lineplot(x='year', y='spch', data = data)
sns.lineplot(x='year', y='acous', data = data)
sns.lineplot(x='year', y='live', data = data)
data['top genre'].value_counts().head(20).plot.pie(figsize=(15,10), autopct='%0.0f%%')