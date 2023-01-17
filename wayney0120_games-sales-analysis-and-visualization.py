import pandas as pd

import itertools as it

import collections as co

import matplotlib.pyplot as plt

from matplotlib import cm

import seaborn as sns

import numpy as np

video = pd.read_csv("../input/videogamesales/vgsales.csv", header=0, index_col=0, encoding='utf-8')
print(video.info())
genre = video.groupby("Genre").aggregate({"Genre": "count"}).rename(columns={'Genre': 'count', 'index': 'Genre'})

genre = genre.sort_values('count', ascending=False).reset_index()

explode = [0.15,0.11,0.09,0.06,0.05,0.04,0.03,0.02,0.01,0.01,0.01,0.01]



plt.figure(figsize=(9, 10))

labels = genre["Genre"]

colors = cm.rainbow(np.linspace(0.3, 0.8, 12))

plt.title(label=" Genres of Games", loc="center", fontsize=15)

plt.pie(genre["count"], autopct='%.1f%%', labels=labels, explode=explode, colors=colors, startangle=20)





plt.show()
platform = video.groupby("Platform").aggregate({"Platform": "count"}).rename(columns={'Platform': 'count', 'index': 'Platform'})

platform = platform.sort_values('count', ascending=False).reset_index()



plt.figure(figsize=(10, 11))

labels = platform["Platform"]

colors = cm.rainbow(np.linspace(0.7, 1, 20))

plt.title(label="Platform of Games", loc="center", fontsize=15)

plt.bar(platform["Platform"].head(10), platform["count"].head(10),

        width=0.5, color=colors)



plt.show()
publisher = video.groupby("Publisher").aggregate({"Publisher": "count"}).rename(columns={'Publisher': 'count', 'index': 'Publisher'})

publisher = publisher.sort_values('count', ascending=False).reset_index()



plt.figure(figsize=(7, 8))

labels = publisher["Publisher"]

colors = cm.rainbow(np.linspace(0.3, 0.5, 6))

plt.title(label="Publishers of Games", loc="center", fontsize=15)

plt.bar(publisher["Publisher"].head(5), publisher["count"].head(5),

        width=0.5, color=colors)

plt.xticks(rotation=20)
year = video.dropna()

year = year.groupby("Year").aggregate({"Year": "count"}).rename(columns={'Year': 'count', 'index': 'Year'})

year = year.sort_values('Year', ascending=False).reset_index()



plt.figure(figsize=(9, 8))

plt.plot(year["Year"], year["count"], color='turquoise')

plt.title(label="Games launched vs Years", loc="center", fontsize=15)

plt.stackplot(year["Year"], year["count"], colors='lightcyan')  # area chart
plt.figure(figsize=(15, 8))

plt.subplot2grid((1,2), (0,0))

plt.title(label="Sales vs Years", loc="center", fontsize=15)

plt.scatter(video["Year"], video["NA_Sales"], color='turquoise', label="NA_Sales")

plt.scatter(video["Year"], video["EU_Sales"], color='wheat', label="EU_Sales")

plt.scatter(video["Year"], video["JP_Sales"], color='lightsteelblue', label="JP_Sales")

plt.scatter(video["Year"], video["Other_Sales"], color='greenyellow', label="Other_Sales")

plt.legend()



plt.subplot2grid((1,2), (0,1))

plt.title(label="Global_Sales vs Years", loc="center", fontsize=15)

glo = video.groupby("Year").aggregate({"Global_Sales": "sum"})

glo = glo.sort_values('Year', ascending=False).reset_index()

plt.plot(glo["Year"], glo["Global_Sales"], color='cyan', label="Global_Sales")

plt.stackplot(glo["Year"], glo["Global_Sales"], colors='lightcyan')  # area chart
GenreGroup = video.groupby(['Genre']).sum().loc[:, 'NA_Sales':'Global_Sales']  

GenreGroup['NA_Sales%'] = GenreGroup['NA_Sales']/GenreGroup['Global_Sales']

GenreGroup['EU_Sales%'] = GenreGroup['EU_Sales']/GenreGroup['Global_Sales']

GenreGroup['JP_Sales%'] = GenreGroup['JP_Sales']/GenreGroup['Global_Sales']

GenreGroup['Other_Sales%'] = GenreGroup['Other_Sales']/GenreGroup['Global_Sales']



plt.figure(figsize=(13, 12))

cmap = sns.cm.rocket_r

ax = sns.heatmap(GenreGroup.loc[:, 'NA_Sales%':'Other_Sales%'],

                 vmax=1, vmin=0, annot=True, fmt='.2%', cmap=cmap)

plt.title("Area_Sales vs Genre")

ax.set_ylim(bottom=12, top=0)



plt.show()
video = video.dropna()

video["Year"] = video["Year"].dropna().astype("int")



table = video.pivot_table('Global_Sales', columns='Year', index='Genre', aggfunc='sum')





plt.figure(figsize=(13, 12))

ax = sns.heatmap(table, cmap='viridis')

plt.title("Sales in according to Genres and Years")

ax.set_ylim(bottom=12, top=0)



plt.show()
from pandas.plotting import table  # table drawing


fig = plt.figure(figsize=(4, 3), dpi=150)



# #Action

rank = video[video['Genre'] == 'Action']

rank = rank.groupby(['Name']).aggregate({"Global_Sales": "sum"})

rank = rank.sort_values(by=['Global_Sales'], ascending=False).reset_index()

rank["Global_Sales"] = rank["Global_Sales"].apply(lambda x: round(x, 2))  # two decimal



ax = fig.add_subplot(211, frame_on=False)

plt.title("Top_action")

ax.xaxis.set_visible(False)  # hide the x axis

ax.yaxis.set_visible(False)  # hide the y axis

table(ax, rank.head(5), loc='center')
fig = plt.figure(figsize=(4, 3), dpi=150)



#Racing

rank2 = video[video['Genre'] == 'Racing']

rank2 = rank2.groupby(['Name']).aggregate({"Global_Sales": "sum"})

rank2 = rank2.sort_values(by=['Global_Sales'], ascending=False).reset_index()

rank2["Global_Sales"] = rank2["Global_Sales"].apply(lambda x: round(x, 2))  # two decimal



ax = fig.add_subplot(212, frame_on=False)

plt.title("Top_racing")

ax.xaxis.set_visible(False)  # hide the x axis

ax.yaxis.set_visible(False)  # hide the y axis

table(ax, rank2.head(5), loc='center')
fig = plt.figure(figsize=(4, 3), dpi=150)



#Role

rank = video[video['Genre'] == 'Role-Playing']

rank = rank.groupby(['Name']).aggregate({"Global_Sales": "sum"})

rank = rank.sort_values(by=['Global_Sales'], ascending=False).reset_index()

rank["Global_Sales"] = rank["Global_Sales"].apply(lambda x: round(x, 2))  # two decimal



ax = fig.add_subplot(212, frame_on=False)

plt.title("Top_Role-Playing")

ax.xaxis.set_visible(False)  # hide the x axis

ax.yaxis.set_visible(False)  # hide the y axis

table(ax, rank.head(5), loc='center')
fig = plt.figure(figsize=(4, 3), dpi=150)



#Shooter

rank2 = video[video['Genre'] == 'Shooter']

rank2 = rank2.groupby(['Name']).aggregate({"Global_Sales": "sum"})

rank2 = rank2.sort_values(by=['Global_Sales'], ascending=False).reset_index()

rank2["Global_Sales"] = rank2["Global_Sales"].apply(lambda x: round(x, 2))  # two decimal



ax = fig.add_subplot(211, frame_on=False)

plt.title("Top_shooter")

ax.xaxis.set_visible(False)  # hide the x axis

ax.yaxis.set_visible(False)  # hide the y axis

table(ax, rank2.head(5), loc='center')