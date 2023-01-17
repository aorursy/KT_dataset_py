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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.head()
df.shape
df.dtypes
df["diagnosis"].value_counts()
#visualize the count

sns.countplot(df.diagnosis,label="count")

plt.show()
sns.pairplot(df.iloc[:,1:6],hue="diagnosis")

plt.show()
# Set the width and height of the figure

plt.figure(figsize=(20,18))



corr = df.corr()

ax = sns.heatmap(corr,vmin=-1,vmax=1,center=0,annot=True)
plt.figure(figsize=(15,8))

sns.distplot(df['radius_mean'], hist=True, bins=30, color='grey')

plt.xlabel('radius_mean')

plt.ylabel('Frequency')

plt.title('Distribution of radius_mean', fontsize=15)
plt.figure(figsize=(15,8))

sns.distplot(df['concavity_mean'], hist=True, bins=30, color='grey')

plt.xlabel('concavity_mean')

plt.ylabel('Frequency')

plt.title('Distribution of concavity_mean', fontsize=15)
plt.figure(figsize=(20,10))

ax = sns.boxplot(data = df, orient = "h", palette = "Set1")

plt.show()