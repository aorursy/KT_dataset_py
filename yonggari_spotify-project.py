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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl



sp = pd.read_csv('../input/top50spotify2019/top50.csv', encoding = 'ISO-8859-1', index_col = 0)



import warnings

warnings.filterwarnings('ignore')



#SEE IF NaN VALUES EXISTS

print(sp.isna().any())



#THIS ONE ADDS UP ALL THE NULL VALUES

print(sp.isnull().sum())



#FILLS THE NaN WITH ZERO

sp.fillna(0)



print(sp.shape)

print(sp.info())

print(sp.head())

print(sp.dtypes)





#RENAME A FEW COLUMNS

sp.rename(columns={'Loudness..dB..':'Loudness(dB)','Valence.':'Valence', 'Acousticness..':'Acousticness','Speechiness.':'Speechiness'},inplace=True)









#HEAT MAP OF OUR DATA

plt.figure(figsize=(10,10))

plt.title('Correlation Map')

ax=sns.heatmap(sp.corr(),

               linewidth=3.1,

               annot=True,

               center=1)















#GENRE BREAKDOWN

print(sp.Genre.head())

print(sp.Genre.describe())



#GIVES TOTAL PER COLUMN FOR EACH GENRE

print(sp.groupby("Genre").sum())



#CALCULATING THE NUMBER OF SONGS OF EACH GENRE

print(sp.groupby('Genre').size().unique)



#PIE CHART OF GENRE BREAKDOWN

import plotly.express as px

from plotly.offline import plot

fig = px.pie(sp, values = 'Popularity', names='Genre', hole = 0.3)

fig.update_layout(annotations=[dict(text='Genre',font_size=20, showarrow=False)])

plot(fig)



















#ARTIST BREAKDOWN

print(sp.groupby('Artist.Name').size())



#BAR GRAPH OF ARTIST VS NUMBER OF SONGS

fig = plt.figure(figsize = (15,7))

sp.groupby('Artist.Name')['Track.Name'].agg(len).sort_values(ascending = False).plot(kind = 'bar')

plt.xlabel('Artist Name', fontsize = 20)

plt.ylabel('Count of songs', fontsize = 20)

plt.title('Artist Name vs Count of songs', fontsize = 30)

plt.show()

















#VALENCE BREAKDOWN



#GRAPH THE VALENCE

plt.figure(figsize = (10, 5))

sns.set(style = 'dark')

sns.distplot(sp['Valence'],

             rug=True,

             hist_kws={"histtype": "stepfilled",

                       'linewidth' : 2,

                       'color':'B',

                      'alpha' : 0.4});

plt.title('Valence. Distribution');

























#DISTRIBUTION OF BEATS PER MINUTE FIG SIZE

plt.figure(figsize = (20, 5));

sns.countplot(data = sp, x = 'Beats.Per.Minute', palette = 'spring');

plt.title('Count of Beats Per Minute');











#POPULARITY BREAKDOWN

print(sp.Popularity.describe())



sns.boxplot( y = sp["Popularity"])

plt.show()



#WHAT'S THE MOST POPULAR SONG?

sp_pop = sp.Popularity.max()

print(sp.loc[sp['Popularity'] == sp_pop, 'Track.Name'])



#WHAT'S THE LEAST POPULAR SONG?

sp_poop = sp.Popularity.min()

print(sp.loc[sp['Popularity'] == sp_poop, 'Track.Name'])



#HOW MANY HAVE A POPULARITY OF 90 AND ABOVE?

sp_90 = sp[sp["Popularity"] >= 90]

print(sp_90.count())



#MOST POPULAR SONGS SIDE BAR GRAPH

sns.set(style = 'whitegrid')

plt.figure(figsize = (6, 30))

sns.barplot(data = sp, y = 'Track.Name', x= 'Popularity');
