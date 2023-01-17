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
# The dataset I have contains video game sales from 1980 up until 2020. The games that are included have all sold over 100,000 copies, and they are seperated by each area in the world, North America, Europe, Japan, and other. The numbers they are labeled by is by the million. 
import pandas as pd
df = pd.read_csv('../input/videogamesales/vgsales.csv')
# First I want to take a look at the data at hand and see what all I have. I noticed we have 11 columns, with a total of 16,598 entries.
df.info()
df.shape
# Getting a better look at the dataset up close I can see there is a number ranking for each game that is based on the overall global sales of the game. I can also see how much the game sold in other parts of the world. It also contains the genre, publisher, what platform the game is played on, when the game was released and the name of the game. 
df.head(5)
# Next I want to find out if there are any null values in my data set, and here I can see there is null in the year and publisher categories.  
df.isnull().sum()
# Since I have such a large data set with so many values, I am just going to go ahead and remove all the entries with null values. 
df_filtered = df.dropna()
# And now double checking just to make sure that they were all removed. 
df_filtered.shape
# Next I want to start exploring the data a little bit more. So i start to look closer at all the specific entries that are in each column, and I started with the 'Platform' I noticed there are 31 different platforms. 
df_filtered['Platform'].nunique()
platform_counts = df_filtered['Platform'].value_counts()
platform_counts
# Then I started looking at the genre and noticed that there are not a ton but they are all very dense, so I think this would be a good topic to dive a little bit deeper into. 
genre_counts = df_filtered['Genre'].value_counts()
genre_counts
# Now I want to look at the years, I want to be able to determine that this data is current and up to date with all the newest games that have been released. Quickly I see that this data set only appears to be valid up until 2016, having only 4 games after 2016 (which I know is not an accurate reading) 
years = df_filtered['Year'].value_counts()
years
# So I am going to go ahead and just remove those years, because I am going to be breaking down by year a little more in depth, I don't want those years to skew the data because it is not up to date. 
new_year_games = df_filtered['Year'] >= 2017 
old_games = new_year_games == False
# this below will tell me what games were made after 2017 and that I do not have an updated data set
df_filtered.iloc[new_year_games.values].head()
df_filtered = df_filtered.iloc[old_games.values]
# this is to double check to make sure that the 4 games were removed from the data set. 
df_filtered.shape
# Now I am going to go ahead and start analyzing the different areas a little closer. I want to determine what is the most profitable area throughout the country. By running some descriptive statistics I can determine that North America has the most profitable area of the dataset. 
df_filtered[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].describe()
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
# next I want to get in depth and analyze the last 2 years, I want to be able to see if there has been any significant change in the overall profits of the last 2 years. Doing this will help me determine how frequently changes happen when buying video games. 
released_2015 = df_filtered.iloc[(df_filtered['Year'] == 2015).values] 
released_2015.info()
released_2016 = df_filtered.iloc[(df_filtered['Year'] == 2016).values]
released_2016.info()
from scipy import stats
# Now that I have my 2 years seperated out, I am going to run some t tests on the global sales of those 2 years. I am going to do the overall global sales amongst all genres because I want an overall look to see how frequently and how much this data is changing. 
stats.ttest_ind(released_2015['Global_Sales'], released_2016['Global_Sales'])
sns.distplot(released_2015['Global_Sales'], bins=30)
sns.distplot(released_2016['Global_Sales'], bins=30)
# With these we can tell the large difference in the overall global sales, and the number of global sales in 2016 vs 2015. 
# now i want to better understand with a visual exactly how different the overall top 50 games are doing compared to one another. I am going to create a scatter plot to show the difference and I am going to arrange my key by genre because my end goal is going to be to help determine what the next best video game genre is going to be
plt.figure(figsize = (18, 7))
sns.scatterplot(x="Rank", y="Global_Sales", s=125, hue="Genre", data=df_filtered.head(50))
plt.xlabel('Ranking')
plt.ylabel('Global Sales (millions)')
plt.show()
# now I want to look at the correlations between the different areas sales and how they relate to one another.
df_filtered[['NA_Sales', 'JP_Sales', 'EU_Sales', 'Other_Sales', 'Global_Sales']].corr()
# because my end goal is to find the next genre of video game i want to seperate the genres out and figure out the mean of each based on location. With this i can single down where certain genres are more popular than others 
df_filtered.groupby(['Genre']).mean()
# lastly what I want to do to help me better determine my end goal i want to know how popular each genre was throughout each decade going back to 1980. With this it will help me better understand what games have been, and still are popular today. 
last_six_years = df_filtered.iloc[(df_filtered['Year'] >= 2010).values]
last_six = last_six_years['Genre'].value_counts()
last_six
the_early_2000 = df_filtered.iloc[((df_filtered['Year'] < 2010) & (df_filtered['Year'] >= 2000)).values]
early_2000 = the_early_2000['Genre'].value_counts()
early_2000
the_1990s = df_filtered.iloc[((df_filtered['Year'] < 2000) & (df_filtered['Year'] >= 1990)).values]
the_1990 = the_1990s['Genre'].value_counts() 
the_1990
the_1980s = df_filtered.iloc[((df_filtered['Year'] < 1990) & (df_filtered['Year'] >= 1980)).values]
the_1980 = the_1980s['Genre'].value_counts()
the_1980
# Comparing all of the number of genres that were popular by each decade, we can predict what games have been/ and will continue to be popular in the future. The genres that are most popular are very similar across every year. 