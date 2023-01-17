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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data= pd.read_csv(os.path.join(dirname, filename))
data.head(3)
data.columns
data.info()
data.set_index('Unnamed: 0', inplace=True)
data.drop('type',axis=1, inplace=True)
#change Rotten Tomatoes to float type which can be plotted as numeric
data['Rotten Tomatoes'] = data['Rotten Tomatoes'].str.replace(r'\D', '').astype(float)
#change IMDb rating to the same scalling as Rotten Tomatoes
data['IMDb']= data['IMDb'].map(lambda x:x*10)
data.head(10)
data.sort_values(by='IMDb', ascending= False).head(10)
data.sort_values(by='Rotten Tomatoes', ascending= False).head(10)

plt.figure(figsize=(20,8))
sns.countplot(x=data[data['Age'].notnull()]['Age'],data=data,palette='rainbow',
              order =data[data['Age'].notnull()]['Age'].value_counts().index)
plt.title('Count vs Age')
plt.figure(figsize=(10,4))
sns.distplot(data[data['Rotten Tomatoes'].notnull()]['Rotten Tomatoes'],
             bins =30, kde=False, color='b',hist_kws=dict(edgecolor="k", linewidth=2))

plt.figure(figsize=(10,4))
sns.distplot(data[data['IMDb'].notnull()]['IMDb'],
            bins=30, color='g',hist_kws=dict(edgecolor="k", linewidth=2) )
plt.figure(figsize=(50,20))
sns.countplot(x=data[data['IMDb'].notnull()]['IMDb'],data=data, color='r')
sns.countplot(x=data[data['Rotten Tomatoes'].notnull()]['Rotten Tomatoes'],data=data,color='b')

a=data[data['Netflix']==1].shape[0]
b=data[data['Hulu']==1].shape[0]
c=data[data['Prime Video']==1].shape[0]
d= data[data['Disney+']==1].shape[0]
total_length= len(data)
data2 = pd.DataFrame([a,b,c,d],index=['Netflix','Hulu','Prime Video','Disney+'], columns=['channel'])
data2
# number of unique tv show title is lesser than total count from sum of different streaming plotform, there were show available in multiples platform
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
tv_platform=pd.melt(data[['Title','Netflix','Hulu','Disney+','Prime Video']],id_vars=['Title'],var_name='StreamingOn',
                      value_name='Present')

tv_platform = tv_platform[tv_platform['Present'] == 1]
tv_platform.drop(columns=['Present'],inplace=True)

df = tv_platform.merge(data, on='Title', how='inner')
px.scatter(df, x=df['IMDb'],y=df['Rotten Tomatoes'],color=df['StreamingOn'])
plt.figure(figsize=(12,8))
sns.countplot(x='Age', hue='StreamingOn', data=df)
plt.legend(loc=1)
