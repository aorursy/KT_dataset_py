import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")#It is a csv file so I used .read_csv()

df.info() #It shows number and types of each features
#convert yaer into date time format

import datetime



df['Year'] = pd.to_datetime(df['Year'], format = '%Y')
df.info()
#highest selling Publisher in Japan

jpop = df.groupby('Publisher')['JP_Sales'].sum()

jpop.sort_values(ascending=False, inplace=True)

top10 = jpop.iloc[:10]

sns.barplot(y = top10.index, x = top10.values, palette="plasma")

plt.ylabel = "Publisher"

plt.xlabel = "Sales in Japan"

plt.show()
ax = df[(df['Publisher']=='Nintendo')].plot(x='Year', y='JP_Sales', figsize=(15,6))

plt.title("Year on Year sales for Nintendo")

plt.show()
ax = df[(df['Publisher']=='Namco Bandai Games')].plot(x='Year', y='JP_Sales', figsize=(15,6))

plt.title("Year on Year sales for Namco Bandai Games")

plt.show()
ax = df[(df['Publisher']=='Konami Digital Entertainment')].plot(x='Year', y='JP_Sales', figsize=(15,6))

plt.title("Year on Year sales for Konami Digital Entertainment")

plt.show()
#Further looking at Nintedos sucess in Japan

df1 = df[(df.Publisher=='Nintendo')]

df1.head()
#Highest Selling Nintendo Games

nintop = df1.groupby('Name')['JP_Sales'].sum()

nintop.sort_values(ascending=False, inplace=True)

top10 = nintop.iloc[:10]

top10
#We se from the previous year on year sales plot for Nintendo, 1996 was their best year, which was highets selling game in in 1996

df2 = df1[(df1.Year=='1996')]

ninsix = df2.groupby('Name')['JP_Sales'].sum()

ninsix.sort_values(ascending=False, inplace=True)

top5 = ninsix.iloc[:5]

top5
#Japans all time top 10 games

jname = df.groupby('Name')['JP_Sales'].sum()

jname.sort_values(ascending=False, inplace=True)

top5 = jname.iloc[:10]

sns.barplot(y = top5.index, x = top5.values, palette="plasma")

plt.ylabel = "Game Name"

plt.xlabel = "Highest Selling game in Japan"

plt.show()
#We now want to findout which are highest selling games in Japan from 2015 onwards

start_date = '01-01-2014'

end_date = '01-01-2020'

mask = (df['Year'] > start_date) & (df['Year'] <= end_date)

df3 = df.loc[mask]

df3
#Top selling games in Japan from 2015 till date

jname2 = df3.groupby('Name')['JP_Sales'].sum()

jname2.sort_values(ascending=False, inplace=True)

top5 = jname2.iloc[:10]

sns.barplot(y = top5.index, x = top5.values, palette="plasma")

plt.ylabel = "Game Name"

plt.xlabel = "Highest Selling game in Japan"

plt.show()