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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('/kaggle/input/chicago-crime-detective/Chicago_Crime_Detective.csv',index_col=0)
df.head()
df.describe()
df.info()
df.dtypes.reset_index()
df['Date']= pd.to_datetime(df['Date'])
df.dtypes.reset_index()
# findout the median of Date
lenght=len(df['Date'])
sorted(df['Date'])[lenght//2]
df['Hour'] = df['Date'].apply(lambda time: time.hour)
df['Month'] = df['Date'].apply(lambda time: time.month)
df['Day'] = df['Date'].apply(lambda time: time.dayofweek)
df
sns.countplot(x='Day',data=df, palette='viridis')
byMonth=df.groupby('Month').count()
byMonth
df.groupby([df.Month]).size().plot(kind='barh')
days = ['Monday','Tuesday','Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
df.groupby([df.Day]).size().plot(kind='barh')
plt.ylabel('Days of the week')
plt.yticks(np.arange(7), days)
plt.xlabel('Number of crimes')
plt.title('Number of crimes by day of the week')
plt.show()
byMonth['Arrest'].plot(kind='barh',figsize=(10, 8))
df.groupby('Year').size().plot(figsize=(10, 8))
f=df[df['Year']==2001]
print(len(f[f['Arrest']==True]))
print(len(f['Arrest']))
len(f[f['Arrest']==True])/len(f['Arrest'])
g=df[df['Year']==2007]
print(len(g[g['Arrest']==True]))
print(len(g['Arrest']))
len(g[g['Arrest']==True])/len(g['Arrest'])
df['LocationDescription'].value_counts().head(5)
h=df[df['LocationDescription']=='STREET']
h.groupby('Day').count()