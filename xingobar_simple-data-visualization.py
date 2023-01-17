# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/database.csv')
df.head()
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(df.gender.dropna(),ax=ax)

plt.title('Gender Count')
fig,ax = plt.subplots(figsize=(8,6))



sns.distplot(df.age.dropna(),ax=ax,color='#ef9a9a')

plt.title('Age Distribution')
armed_df = df['armed'].value_counts().sort_values(ascending=False)[:15].reset_index()

fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x=armed_df['index'],y=armed_df['armed'],ax=ax)

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

plt.title('Top 10 Armed')

plt.xlabel('Armed Category')

plt.ylabel('Total Amount of Armed')
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(df.manner_of_death.dropna(),ax=ax)

plt.title('The manner of death')
df.threat_level.unique()
sns.countplot(df.threat_level.dropna())

plt.title('Threat Level')
plt.title('Threat Level by gender')

sns.countplot(df.threat_level.dropna(),hue=df.gender.dropna())
plt.title('The manner of death by gender')

sns.countplot(df.manner_of_death,hue=df.gender)