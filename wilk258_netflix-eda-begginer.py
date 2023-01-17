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
import random
import plotly_express as px
df = pd.read_csv("/kaggle/input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv")
df.head(3)
sns.kdeplot(data=df['IMDb'], shade=True)
print(df.Year.value_counts().sort_values(ascending = False).head(30))
sns.kdeplot(data=df['Year'], shade=True)
df.Title.value_counts().sort_values(ascending = False).head(30)
print(df.Age.value_counts())
sns.countplot(df['Age'], palette = 'hsv')
plt.title('Age distribution', fontsize = 20, fontweight = 100)
plt.xticks(rotation = 90)
plt.show()
#sns.countplot(x='Age', data=df)
print(df.Netflix.value_counts())
sns.countplot(x='Netflix', data=df)
print(df.Hulu.value_counts())
sns.countplot(x='Hulu', data=df)
print(df['Prime Video'].value_counts())
sns.countplot(x='Prime Video', data=df)
print(df['Disney+'].value_counts())
sns.countplot(x='Disney+', data=df)
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)

plt.title('Correlogram of netfix', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
from wordcloud import WordCloud

wc = WordCloud(background_color = 'lightgray',
              width = 2000,
              height = 2000,
              colormap = 'magma',
              max_words = 70).generate(str(df['Title']))

plt.rcParams['figure.figsize'] = (15, 15)
plt.title('Neflix Title wordcount', fontsize = 30, fontweight = 10)

plt.imshow(wc)
plt.axis('off')

plt.show()

