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
df = pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')
df.head()
df.info()
df['date']=pd.to_datetime(df['date'])
df.columns
df.shape
df.isnull().sum()
df.dropna(inplace=True)
df.shape
sns.countplot(df['gender'])
sns.distplot(df['age'],kde = False)
df.columns
import wordcloud
from wordcloud import WordCloud
wordcloud1 = WordCloud().generate(' '.join(df['armed']))
armed=list(df['armed'].dropna().unique())

fig, (ax2) = plt.subplots(1,1,figsize=[17, 10])

wordcloud2 = WordCloud().generate(" ".join(armed))

ax2.imshow(wordcloud2,interpolation='bilinear')

ax2.axis('off')

ax2.set_title('Most Used Arms',fontsize=20)
sns.countplot(df['manner_of_death'])
df['flee'].value_counts()
df['flee'].count()
print("We See {}% targets don't flee.".format(round(2965*100/4399),2))
print(f'{len(df.loc[(df.flee=="Not fleeing") & (df.armed=="unarmed")])} cases were unarmed and did not flee. Yet thee targets were killed.')
print(f'In {len(df.loc[(df.body_camera==False) & (df.armed=="unarmed")])} cases, the target was unarmed and the cop had NO body camera.')
print(f'In {len(df.loc[(df.race=="B") & (df.manner_of_death=="shot")])} cases, the target was black and was shot.')
df['manner_of_death'].value_counts()
print("We See {}% targets were black and got shot.".format(round(1100*100/1469),2))
sns.countplot(df['race'])
import plotly.express as px

import plotly.graph_objects as go
fig = go.Figure(go.Bar(

    x= df.groupby('city').agg('count')['id'].sort_values(ascending=False)[:20].index, 

    y= df.groupby('city').agg('count')['id'].sort_values(ascending=False)[:20].values,  

    text=df.groupby('city').agg('count')['id'].sort_values(ascending=False)[:20].index,

    textposition='outside',

    marker_color=df.groupby('city').agg('count')['id'].sort_values(ascending=False)[:20].values

))

fig.update_layout(title='Shootout by City Stats')

fig.show()