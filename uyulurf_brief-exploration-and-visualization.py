# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import seaborn as sns

import os

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/top50spotify2019/top50.csv", encoding='ISO-8859-1')

df.head()
df = df.drop(columns=['Unnamed: 0'])

df.shape
corr = df.corr()

sb.heatmap(corr)
sns.pairplot(df, kind='reg')
sns.lmplot( x="Beats.Per.Minute", y="Speechiness.", data=df, fit_reg=False, hue='Genre', legend=True)

autopct=('%1.1f%%')

labels=df["Genre"].value_counts().index

sizes=df["Genre"].value_counts().values

plt.figure(figsize = (10,10))

plt.pie(sizes, labels=labels)

autopct=('%1.1f%%')
