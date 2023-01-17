# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as ws
ws.filterwarnings ("ignore")
sns.set_style("white")
df = pd.read_csv("/kaggle/input/world-top-chess-players-august-2020/top_chess_players_aug_2020.csv")
df.head()
# Selecting players with Standard_Rating >2000
df = df[df['Standard_Rating']>2000]
top_10_players = df.head(10)
top_10_players
# Rating distribution of Top 10 Chess Players
plt.figure(figsize=(12,6))
plt.title("Top 10 Chess Players")
sns.barplot(x = "Standard_Rating", y = "Name", data=top_10_players).set_xlim(2750, 2870)
plt.show()
top_10_women_players = df[df.Gender=='F'].head(10).reset_index(drop=True)
top_10_women_players
# Rating distribution of Top 10 Women Chess Players
plt.figure(figsize=(12,6))
plt.title("Top 10 Women Chess Players")
sns.barplot(x = "Standard_Rating", y = "Name", data=top_10_women_players).set_xlim(2500, 2680)
plt.show()
print("Player's Gender Distribution")
print(df.Gender.value_counts())

# Pie chart of Gender distribution on Chess Players
labels = ['Male', 'Female']
sizes = df.Gender.value_counts()
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(7,7))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', colors=['lightskyblue', 'peachpuff'])
plt.show()
print("Grandmaster's Gender Distribution")
gms = df[df.Title=='GM']
print(gms.Gender.value_counts())

# Pie chart of Gender distribution on Chess Players
labels = ['Male', 'Female']
sizes = gms.Gender.value_counts()
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(7,7))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', colors=['lightskyblue', 'peachpuff'])
plt.show()
# Average age of the chess players
birth_year = df.Year_of_birth.values
age_value =  2020 - birth_year
print("Average age of chess players is", round(np.nanmean(age_value), 1), "years.")
title_dist = df.Title.value_counts().reset_index()
print(title_dist)
plt.figure(figsize=(8,5))
plt.title("Title distribution of Players")
sns.barplot(x = "index", y = "Title", data = title_dist)
plt.show()
avg_std_rating_per_title = round(df.groupby("Title")["Standard_Rating"].mean(), 2).reset_index().sort_values(by='Standard_Rating', ascending=False).reset_index(drop=True)
print(avg_std_rating_per_title)
plt.figure(figsize=(9,5))
plt.title("Average Standered rating of player as per title")
sns.barplot(x = "Title", y="Standard_Rating", data=avg_std_rating_per_title, palette="Reds_d").set_ylim(1800, 2525)
plt.show()
avg_rapid_rating_per_title = round(df.groupby("Title")["Rapid_rating"].mean(), 2).reset_index().sort_values(by='Rapid_rating', ascending=False).reset_index(drop=True)
print(avg_rapid_rating_per_title)
plt.figure(figsize=(9,5))
plt.title("Average Rapid rating of player as per title")
sns.barplot(x = "Title", y="Rapid_rating", data=avg_rapid_rating_per_title, palette="Blues_d").set_ylim(1800, 2500)
plt.show()
avg_blitz_rating_per_title = round(df.groupby("Title")["Blitz_rating"].mean(), 2).reset_index().sort_values(by='Blitz_rating', ascending=False).reset_index(drop=True)
print(avg_blitz_rating_per_title)
plt.figure(figsize=(9,5))
plt.title("Average Blitz rating of player as per title")
sns.barplot(x = "Title", y="Blitz_rating", data=avg_blitz_rating_per_title, palette="Greens_d").set_ylim(1800, 2525)
plt.show()
countries_dist = df.Federation.value_counts().reset_index().rename(columns={'index':'Country', 'Federation':'Total players'})[:10]
countries_dist.index += 1
print(countries_dist)

# Pie chart of Country-wise distribution of Chess Players
labels = countries_dist['Country']
sizes = countries_dist['Total players']
fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Country-wise Distribution of Chess Players")
plt.show()
countries_dist = df[df.Title=='GM'].Federation.value_counts().reset_index().rename(columns={'index':'Country', 'Federation':'Total GMs'})[:10]
print(countries_dist)
plt.figure(figsize=(10,5))
plt.title("Country-wise Distribution of GMs")
sns.barplot(x = "Country", y="Total GMs", data=countries_dist)
plt.show()
countries_dist = df[df.Standard_Rating>2700].Federation.value_counts().reset_index().rename(columns={'index':'Country', 'Federation':'Super GMs'})[:10]
print(countries_dist)
plt.figure(figsize=(10,5))
plt.title("Country-wise Distribution of Super GMs")
sns.barplot(x = "Country", y="Super GMs", data=countries_dist)
plt.show()