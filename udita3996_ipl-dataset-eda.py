import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import glob

import warnings

warnings.filterwarnings("ignore")
path = '/kaggle/input/indian-premier-league-20082019/IPL_2008_2019' 

all_files = glob.glob(path + "/*.csv")



li = []



for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)

    li.append(df)



match_data = pd.concat(li, axis=0, ignore_index=True)
match_data.shape
match_data.isnull().sum()
match_data.info()
match_data.groupby("Batting_Team").Runs.sum().sort_values(ascending=False).plot.bar()

plt.xlabel("Team")

plt.ylabel("Total Runs")

plt.show()
plt.figure(figsize=(20,10))

top_scorers = match_data.groupby("Player").Runs.sum().sort_values(ascending=False).head(50)

top_scorers.plot.bar()

plt.ylabel("Runs Scored",fontsize=12)

plt.title("Top 50 batsmen",fontsize=15)

plt.show()
match_data["Four"] = match_data.Runs.apply(lambda x : 1 if x==4 else 0)

match_data.head()
match_data["Six"] = match_data.Runs.apply(lambda x : 1 if x==6 else 0)

match_data.head()
plt.figure(figsize=(20,10))

match_data.groupby("Player").Four.sum().sort_values(ascending=False).head(50).plot.bar()

plt.ylabel("Number of Fours",fontsize=12)

plt.title("Top 50 players to hit maximum Fours",fontsize=15)

plt.show()
plt.figure(figsize=(20,10))

match_data.groupby("Player").Six.sum().sort_values(ascending=False).head(50).plot.bar()

plt.ylabel("Number of Sixes",fontsize=12)

plt.title("Top 50 players to hit maximum Sixes",fontsize=15)

plt.show()
match_data["Bowl_Count"] = 1

match_data.head()
bowler_data = match_data.groupby("Baller").sum()

bowler_data.drop(["Inning","Over","Four","Six"],axis = 1,inplace = True)

bowler_data.head()
bowler_data.shape
#Dropping bowlers with less than 10 overs [i.e, Bowl_Count <= 60]

bowler_data = bowler_data[bowler_data.Bowl_Count > 60]

bowler_data.shape
bowler_data["Economy_Rate"] = ((bowler_data["Runs"]+bowler_data["Extra"])/bowler_data["Bowl_Count"])*6

bowler_data.head()
bowler_data.sort_values("Economy_Rate",inplace = True)

bowler_data.reset_index(inplace=True)

bowler_data.head()
#Keep only top 100 bowlers

cutoff_EconomyRate = bowler_data.iloc[99]["Economy_Rate"]

bowler_data = bowler_data[bowler_data["Economy_Rate"]<=cutoff_EconomyRate]

bowler_data.shape
top_bowlers = list(bowler_data.Baller.head(50))

len(top_bowlers)
plt.figure(figsize=(20,10))

sns.barplot(data = bowler_data,x="Baller",y="Economy_Rate")

plt.xticks(rotation=90)

plt.title("Top 100 Bowlers",fontsize=15)

plt.show()
bat_bowl = pd.pivot_table(data = match_data, index = "Player", columns = "Baller", values = "Runs",aggfunc = np.sum,fill_value = 0.0)

bat_bowl.shape
#Get all the bowlers that are not in the top 50 bowlers list

bowlers = list(bat_bowl.columns)

drop_bowlers = []

for bowler in bowlers :

    if bowler not in top_bowlers :

        drop_bowlers.append(bowler)
#Drop all bowlers apart from the top 50 bowlers

bat_bowl.drop(columns = drop_bowlers,inplace = True)

bat_bowl.shape
#Get all the batsmen that are not in the top 50 batsmen list

batsmen = list(bat_bowl.index)

drop_batsmen = []

for batsman in batsmen :

    if batsman not in top_scorers :

        drop_batsmen.append(batsman)
bat_bowl.drop(drop_batsmen,inplace = True)

bat_bowl.shape
plt.figure(figsize=(25,15))

sns.heatmap(data = bat_bowl,center = 80,linewidths = .2)

plt.title("Top 50 Batsmen vs Top 50 Bowlers",fontsize=15)

plt.show()