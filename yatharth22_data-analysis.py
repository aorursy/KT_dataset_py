import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt



%matplotlib inline



la_liga = pd.read_csv("../input/LaLiga_dataset.csv")

season_winner_points = la_liga.groupby(["season"], as_index=False)[["season","points"]].apply(lambda x: x.max())

list_of_champions = []



for i in range(len(season_winner_points["season"])):

    list_of_champions.append(list(la_liga[(la_liga["season"] == str(season_winner_points["season"][i])) & (la_liga["points"] == int(season_winner_points["points"][i]))]["club"].values)[0])



season_winner_points["champions"] = list_of_champions

season_winners = season_winner_points.groupby("champions")["champions"].apply(lambda x: x.count()).sort_values(ascending=False)

ax = season_winners.plot(kind="bar", title="Most number of titles", color="red", alpha=0.7);

for index,data in enumerate(season_winners):

    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=10))

ax.set_xlabel("Club",fontsize=12);

ax.set_ylabel("Number of Titles",fontsize=12);

ax.set_ylim(0,25);