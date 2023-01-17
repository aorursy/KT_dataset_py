# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/hospitals-count-in-india-statewise/Hospitals count in India - Statewise.csv')
df.rename(columns={'States/UTs': 'states','Number of hospitals in public sector': 'public_hospitals', 'Number of hospitals in private sector':
                   'private_hospitals', 'Total number of hospitals (public+private)': 'total_hospitals'}, inplace=True)
df.info()
df['total_hospitals'] = pd.to_numeric(df['total_hospitals'], errors='coerce')
df['public_hospitals'] = pd.to_numeric(df['public_hospitals'], errors='coerce')
df['private_hospitals'] = pd.to_numeric(df['private_hospitals'], errors='coerce')
df.head()
#df['total_hospitals'].plot.hist()
df.info()

# Let's see Total hospitals status
df_p=df.head(100)
plt.figure(figsize=(7,7))
sns.set(style="darkgrid")
ax = sns.countplot(y="total_hospitals", data=df_p.head(200))
# how about Private hospitals status
df_p=df.head(100)
plt.figure(figsize=(7,7))
sns.set(style="darkgrid")
ax = sns.countplot(y="private_hospitals", data=df_p.head(200))
# how about Public hospitals
df_p=df.head(100)
plt.figure(figsize=(7,7))
sns.set(style="darkgrid")
ax = sns.countplot(y="public_hospitals", data=df_p.head(200))
df_c=df.groupby(['states']).count()
df_c.info()
df_c=df.head(100)
plt.figure(figsize=(7,7))
sns.set(style="darkgrid")
ax = sns.countplot(y="states", data=df_p.head(100))
# Count total hospitals
df_p=df.head(100)
plt.figure(figsize=(7,7))
sns.set(style="darkgrid")
ax = sns.countplot(y="total_hospitals", data=df_p.head(200))
bins=range(2,100,15)
plt.hist(df_p["states"],bins,histtype="bar",rwidth=1.0,color='G')
plt.xlabel('states') #set the x label name
plt.ylabel('Count') #set the y label name
plt.plot()
plt.show()
df['private_hospitals'].plot.hist(bins=20,edgecolor='y').autoscale(enable=True,axis='both',tight=True)
df.plot.barh()
df.plot.bar(stacked=True)
# Ratio of hospitals
df.plot.line(y=['public_hospitals','private_hospitals'],figsize=(10,4),lw=4)
df.plot.scatter(x='states',y='total_hospitals')
df.plot.scatter(x='states',y='private_hospitals',c='private_hospitals',cmap='coolwarm')
df['total_hospitals'].plot.kde()
df.plot.scatter(x='private_hospitals',y='public_hospitals',c='g',figsize=(12,3),s=3)
df[['total_hospitals','states']].plot(figsize=(12,5))
df['public_hospitals'].plot(figsize=(12,5))
df['public_hospitals'].plot(figsize=(12,5))
df.rolling(window=15).mean()['private_hospitals'].plot()
df['total_hospitals'].expanding().mean().plot(figsize=(12,5))
df['private_hospitals'].plot.line(figsize=(10,3),ls=':',c='g',lw=2)
title = "India Hospital Distribution"
xlabel = 'Public hospitals'
ylabel = 'Private hospitals'
ax = df['total_hospitals'].plot.line(figsize=(10,4),ls=':',c='b',lw=3,title=title)
ax.set(xlabel=xlabel, ylabel=ylabel)
df['total_hospitals'].plot(figsize=(12,5))
df['total_hospitals'].plot.hist(grid=True)
df.columns
# to be continued...