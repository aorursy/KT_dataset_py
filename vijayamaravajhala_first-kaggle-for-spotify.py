# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot  as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the csv file into a dataframe

df=pd.read_csv("/kaggle/input/top50spotify2019/top50.csv",encoding='ISO-8859-1')

df.head()
# Drop the Unnamed:0 column

df.drop('Unnamed: 0', axis = 1,inplace=True )

df.head()
# Renaming the columns

df.rename(columns={'Track.Name':'track_name','Artist.Name':'artist_name','Beats.Per.Minute':'beats_per_minute','Loudness..dB..':'Loudness(dB)','Valence.':'Valence','Length.':'Length', 'Acousticness..':'Acousticness','Speechiness.':'Speechiness'},inplace=True)

df.head()
# Now check the size of rows and columns with the dtypes 

print("Shape of the dataset : ", df.shape)

df.dtypes
# Check if there  is any null values present in the columns

df.isnull().sum()
# To check the counts of the records based on Genres

gen=df.Genre.value_counts()

gen
# As most of the Genres can come under same type ,lets add them to a new group



df['GeneralGenre']=['hip hop' if each =='atl hip hop'

                      else 'hip hop' if each =='canadian hip hop'

                      else 'hip hop' if each == 'trap music'

                      else 'pop' if each == 'australian pop'

                      else 'pop' if each == 'boy band'

                      else 'pop' if each == 'canadian pop'

                      else 'pop' if each == 'dance pop'

                      else 'pop' if each == 'panamanian pop'

                      else 'pop' if each == 'pop'

                      else 'pop' if each == 'pop house'

                      else 'electronic' if each == 'big room'

                      else 'electronic' if each == 'brostep'

                      else 'electronic' if each == 'edm'

                      else 'electronic' if each == 'electropop'

                      else 'rap' if each == 'country rap'

                      else 'rap' if each == 'dfw rap'

                      else 'escape room' if each == 'hip hop'

                      else 'latin' if each == 'latin'

                      else 'r&b' if each == 'r&n en espanol'

                      else 'raggae' for each in df['Genre']]
# Now groupby General Genre and find the counts

df.GeneralGenre.value_counts()
# Plotting the graph 

sns.countplot(x='GeneralGenre',data=df)
##### group by artist_name

top_5=df.artist_name.value_counts().head()

top_5
# Top 5 Artists in Spotify for the year 2019

top_5.plot(kind='bar')

plt.title('Top Artists in Spotify for 2019')

plt.ylabel('Count')

plt.xlabel('Artists')

plt.show()
Category_df=df[['artist_name','Popularity','GeneralGenre','track_name']]

Category_df.head(2)
# Plotting for categorical values of Popularity Vs GeneralGenre

sns.catplot(y='Popularity',x='GeneralGenre',kind='bar',data=Category_df)