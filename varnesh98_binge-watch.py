# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import csv file(data)
data = pd.read_csv('/kaggle/input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv')
#informaion regarding the data
data.info()
#shape of the Dataset
data.shape
#first 5 rows of the dataset
data.head()
#Last 5 rows of the dataset
data.tail()
#dimension of the object
data.ndim
#size of the object
data.size
#columns/ features of the dataset
data.axes
#columns/ features of the dataset
data.columns
#Datatypes of all the columns
data.dtypes
#checking the emptyiness of the datset
data.empty
#Technique to convert DataFrame to Numpy array
dt = data.values
dt[0]
type(dt)
print('Tv shows on Netflix:',data['Netflix'].sum(),'/',data['Netflix'].count())
print('Tv shows on Hulu:',data['Hulu'].sum(),'/',data['Hulu'].count())
print('Tv shows on Prime Video:',data['Prime Video'].sum(),'/',data['Prime Video'].count())
print('Tv shows on Disney+:',data['Disney+'].sum(),'/',data['Disney+'].count())
#describe the dataset with some basic functionality
data.describe(include='all')
#top 50 IMDb rated Tv shows
plt.subplots(figsize=(10,20))
sns.barplot(x="IMDb", y="Title" , data= data.sort_values("IMDb",ascending=False).head(50))
data.plot.scatter(x='IMDb', y='Year')
#quantity of shows on various platforms present in the dataset
labels = 'Netflix' , 'Hulu', 'Prime Video', 'Disney+'
sizes = [data['Netflix'].sum(),data['Hulu'].sum(),data['Prime Video'].sum(),data['Disney+'].sum()]
explode = (0.1, 0.1, 0.5, 0.1 )

fig1 , ax1 = plt.subplots()

ax1.pie(sizes,
        explode = explode,
        labels = labels,
        autopct = '%1.1f%%',
        shadow = True,
        startangle = 100)

ax1.axis ('equal')
plt.show()
netflix_shows = data.loc[data['Netflix'] == 1]
hulu_shows = data.loc[data['Hulu'] == 1]
prime_video_shows = data.loc[data['Prime Video'] == 1]
disney_shows = data.loc[data['Disney+'] == 1]
#list of top shows on netflix
netflix_top_shows = netflix_shows.loc[netflix_shows['IMDb']>8.0]
hulu_top_shows = hulu_shows.loc[hulu_shows['IMDb']>8.0]
prime_video_top_shows = prime_video_shows.loc[prime_video_shows['IMDb']>8.0]
disney_top_shows = disney_shows.loc[disney_shows['IMDb']>8.0]
#lets plot a bar graph of platforms with highest IMDb shows
platform = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
count = [netflix_top_shows['IMDb'].sum(),hulu_top_shows['IMDb'].sum(),prime_video_top_shows['IMDb'].sum(),disney_top_shows['IMDb'].sum()]
fig = plt.figure()
ax = fig.add_axes([1,1,1,1])
ax.bar(platform,count)
plt.show()

#Platform with most shows rated above 8+ (IMDB)
plt.figure(figsize = (15, 10))
sns.barplot(
           x = platform,
           y = count
)
plt.xlabel('Platform')
plt.ylabel('Total number of showsrated above 8')
plt.title('Platform with most shows rated above 8+ (IMDB)')
plt.show()

#lets plot a bar graph of years with highest IMDb shows
top1990_shows = data.loc[(data['IMDb'] >= 8.0) & (data['Year']<= 1990)]
top2000_shows = data.loc[(data['IMDb']>=8.0) & (data['Year']>1990) & (data['Year']<=2000)]
top2010_shows = data.loc[(data['IMDb']>=8.0) & (data['Year']>2000)&(data['Year']<=2010)]
top2020_shows = data.loc[(data['IMDb']>=8.0) & (data['Year']>2010)&(data['Year']<=2020)]
years = ['< 1990', '1990 - 2000', '2001-2010', '2011-2020']
counts = [top1990_shows['IMDb'].sum(),top2000_shows['IMDb'].sum(),top2010_shows['IMDb'].sum(),top2010_shows['IMDb'].sum()]

plt.figure(figsize = (15, 10))
sns.barplot(
           x = years,
           y = counts
)
plt.xlabel('Years')
plt.ylabel('Total number of showsrated above 8')
plt.title('Years with most shows rated above 8+ (IMDB)')
plt.show()
all_rated = data.loc[data['Age']=='all']
_16_rated = data.loc[data['Age']=='16+']
_18_rated = data.loc[data['Age']=='18+']
print(len(all_rated))
print(len(_16_rated))
print(len(_18_rated))


age = ['Shows for all', '16+', '18+']
counts = [len(all_rated),len(_16_rated),len(_18_rated)]

plt.figure(figsize = (15, 10))
sns.barplot(
           x = age,
           y = counts
)
plt.xlabel('Age')
plt.ylabel('Count of Tv Shows under age restrcition')
plt.title('Age restriction')
plt.show()
# movie with IMDb 8+ which are for all age groups
all_rated_high_rate = all_rated.loc[all_rated['IMDb']>=8.0]
rated_16_high_rate = _16_rated.loc[_16_rated['IMDb']>=8.0]
rated_18_high_rate = _18_rated.loc[_18_rated['IMDb']>=8.0]
#Top IMDb rated Shows analysed on basis of age group
age = ['Shows for all', '16+', '18+']
counts = [len(all_rated_high_rate),len(rated_16_high_rate),len(rated_18_high_rate)]

plt.figure(figsize = (15, 10))
sns.barplot(
           x = age,
           y = counts
)
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Top IMDb rated shows analysed on basis of age group')
plt.show()
#Must watch shows
must_watch = data.loc[data['IMDb']>9.0]
must_watch['Title']
#Top 5 Must watch Shows
a = data.sort_values("IMDb",ascending=False).head(5)
a['Title']