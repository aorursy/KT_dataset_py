import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as ws
ws.filterwarnings ("ignore")
sns.set(font_scale=1.2)
sns.set_style("white")
df = pd.read_csv("/kaggle/input/top-women-chess-players/top_women_chess_players_aug_2020.csv")
df.head()
df.info()
df.shape
top_ten_players = df[:10]
top_ten_players
plt.figure(figsize=(12,6))
plt.title("Standard Rating of Top 10 Women Chess Players")
sns.barplot(x = "Standard_Rating", y = "Name", data=top_ten_players).set_xlim(2400, 2700)
plt.show()
title_dist = df.Title.value_counts().reset_index()
title_dist
plt.figure(figsize=(10,6))
plt.title("FIDE Title distribution of Players")
sns.barplot(x = "index", y = "Title", data = title_dist)
plt.show()
gms = df[df.Title=='GM'].reset_index()
gms
print("There are total {} female chess Grandmasters (GMs) in world currently.".format(len(gms)))
countries_dist = df[df.Title=='GM'].Federation.value_counts().reset_index().rename(columns={'index':'Country', 'Federation':'Total GMs'})[:10]
countries_dist
plt.figure(figsize=(10,6))
plt.title("Country-wise Distribution of Grandmasters")
sns.barplot(x = "Country", y="Total GMs", data=countries_dist)
plt.show()