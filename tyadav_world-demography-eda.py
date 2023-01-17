# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/world-population-19602018/pop_worldometer_data.csv')
df.head()
# Let's check the overall population
df['Population (2020)'].plot.hist()
df.plot.barh()
# Let's check the fertility
df['Fert. Rate'].plot.hist(grid=True)
# Ratio of Population and Fertility rate
df.plot.line(y=['Country (or dependency)','Population (2020)','Fert. Rate'],figsize=(10,4),lw=4)
df.plot.scatter(x='Country (or dependency)',y='World Share %')
# Let's check the Urban population in respect of Median Age, countrywise
df.plot.scatter(x='Country (or dependency)',y='Urban Pop %',s=df['Med. Age']*50,alpha=0.3)
df['Med. Age'].plot.kde()
df.plot.scatter(x='Yearly Change %',y='Net Change')
df.plot.hexbin(x='Density (P/Km²)',y='Land Area (Km²)',gridsize=25,cmap='coolwarm')
ax = df.plot.line(figsize=(12,6))
ax.legend(loc=1) #,bbox_to_anchor=(1.0,1.0))
df.plot.scatter(x='Urban Pop %',y='World Share %',c='g',figsize=(15,5),s=3)
# Population yearly change
df['Yearly Change %'].plot.hist()
# Let's recereate the histogram by tightening the x-axis and adding lines between bars
df['Yearly Change %'].plot.hist(edgecolor='y').autoscale(axis='x',tight=True)
# Let's check the bleanded area as per various parameters
df.loc[0:30].plot.area(stacked=False,alpha=0.3)
ax = df.loc[0:30].plot.area(stacked=False,alpha=0.4)
ax.legend(loc=1,bbox_to_anchor=(1.3,0.5))
# How is Fertility ratio
df['Fert. Rate'].plot(figsize=(12,5))
# Let's check the Urban population 
df['Urban Pop %'].plot(figsize=(12,5))
df.rolling(window=15).mean()['Urban Pop %'].plot()
# Migrants ratio across the world
df[['Migrants (net)','World Share %']].plot(figsize=(12,5))
# How is Urban population percentage
df['Urban Pop %'].expanding().mean().plot(figsize=(12,5))
df['World Share %'].plot.line(figsize=(10,3),ls=':',c='g',lw=2)
# And finally let's check the ratio of migrations 
title = "World migrations"
xlabel = 'Total Migrants'
ylabel = 'Ratio'
ax = df['Migrants (net)'].plot.line(figsize=(12,4),ls=':',c='b',lw=3,title=title)
ax.set(xlabel=xlabel, ylabel=ylabel)
