# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/D2014-18.csv')
df.head()
df.columns = ['Date', 'Time', 'Country', 'Votality', 'Description', 'Evaluation', 'Percentage', 'Actual', 'Forecast', 'Previous Data']
df.head()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,8))
sns.barplot(y = df['Country'].value_counts().index, x = df['Country'].value_counts(), palette = 'summer')
plt.title("Epicenters of Forex events around the world")
plt.xlabel('Number of Events')
df_brexit = df[df['Date']=='2017/03/29']
df_brexit.head()
plt.figure(figsize=(15,8))
sns.barplot(y = df_brexit['Country'].value_counts().index[:7], x = df_brexit['Country'].value_counts()[:7], palette = 'Accent')
plt.title("Economic events on the Brexit Day")
plt.xlabel('Number of Events')
