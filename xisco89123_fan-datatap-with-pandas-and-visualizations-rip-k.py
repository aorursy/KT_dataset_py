import csv

import sklearn

import scipy

import tensorflow as tf

import matplotlib.pyplot as plt

import statsmodels
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input director)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        f_format = filename.split(".")[1]

        fpath = os.path.join(dirname, filename)  

        if "csv" in f_format:

            df_allstar = pd.read_csv(fpath, sep=",")

      

df_allstar.head()
df_allstar.info()
#### Transforming Data one column at a time and expanding dimension space. 

#### regex works here nice to extract exactly the set of characters we wish to match. 

df_allstar["HT"] = df_allstar["HT"].str.replace("-", ".")

df_allstar["NBA Draft Year"] = df_allstar["NBA Draft Status"].str.extract(r"([0-9]{4})")

df_allstar["NBA Draft Round"] = df_allstar["NBA Draft Status"].str.extract(r"(?<=Rnd )(\d)(?=\sPick)").fillna(0)

df_allstar["NBA Draft Pick"] = df_allstar["NBA Draft Status"].str.extract(r"((?<=Pick )\d)").fillna(0)

df_allstar["NBA Drafted"] = df_allstar["NBA Draft Status"].apply(lambda x : 0  if "NBA" in x else 1)

df_allstar.drop(columns=["NBA Draft Status"])

df_allstar["HT"] = df_allstar["HT"].astype(float)

df_allstar["NBA Draft Year"] = df_allstar["NBA Draft Year"].astype(int)

df_allstar["NBA Draft Round"] = df_allstar["NBA Draft Round"].astype(int)

df_allstar["NBA Draft Pick"] = df_allstar["NBA Draft Pick"].astype(int)

df_allstar = df_allstar.drop(columns="NBA Draft Status")
df_allstar.head()
df_participations = df_allstar.groupby(["Player"]).count().sort_values(by="Year", ascending=False)["Year"]

### Plotting in a pandas fashion.

df_participations.reset_index().rename(columns={"Year":"Nr.Participations"})
## We compute average weights per team and per year. We round the number and sort values in a descending order. Only the first 20

## instances(by weight) are printed.

dfAvgWeights = df_allstar.groupby(["Year", "Team"])["WT"].mean().round(2).astype(int).sort_values(ascending=False).reset_index()

dfAvgWeights


#  x["Year"].min()- x["NBA Draft Year"]** to get years from Draft to 1st AllStar

RoadToAllStar = df_allstar.groupby(["Player"]).apply(lambda x: x["Year"].min()- x["NBA Draft Year"].min())

df_final = pd.merge(df_allstar, RoadToAllStar.reset_index(), on="Player", how="inner")

df_final.head()
df_allstar["DraftToAllStar"] = df_final.dropna()[0]

df_allstar = df_allstar.dropna()

df_allstar["DraftToAllStar"] = df_allstar["DraftToAllStar"].astype(int)



df_allstar.head()
position = "G"  # filtering condition/clause

df_allstar.where(df_allstar["Pos"] == position).groupby(["Player"]).count().sort_values(by="Year" ,ascending=False)["Year"].reset_index()

columns = df_allstar.columns.values

clause = (df_allstar["Pos"] == position) 

filtered_df = df_allstar.where(clause).dropna()

GroupingColumns = ["Player", "Selection Type"]

filtered_df.groupby(["Player", "Selection Type"]).count()["Year"].reset_index()

GroupColumns = ["Selection Type", "Player"]

grouped_df = filtered_df.groupby(GroupColumns).count()["Year"].reset_index().rename(columns={"Year": "TotalNrYears"})

grouped_df
coach_selection = df_allstar.where(df_allstar["Selection Type"].str.contains("Coaches")).dropna()

coach_position_selection = coach_selection.groupby(["Pos"]).count().sort_values(by="Year", ascending=False)["Year"].reset_index().rename(columns={"Year":"Total_Selections"})

coach_position_selection
df_allstar["Nationality"].value_counts().reset_index().rename(columns={"index": "Country"}).style.set_properties(**{"text-align": "left"})
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

df_allstar.head()
plt.clf()

data = df_allstar["NBA Drafted"].apply(lambda x : "Not Drafted" if x == 0 else "Drafted").value_counts().to_dict()



array = plt.pie(x=data.values(), labels=data.keys(), explode=(0.5, 0.2), autopct='%1.1f%%', shadow=True, startangle=180)

plt.title("Drafted vs Non-Drafted Players between 2000-2016")



wedges = array[0]

wedges[1].set_animated(True)
plt.clf()



df = df_allstar["NBA Draft Pick"].value_counts().reset_index().rename(columns={"index": "Draft Pick", "NBA Draft Pick": "TotalNrPlayers"})

df = df[(df["Draft Pick"] != 0)]



scalars = df["Draft Pick"]

height = df["TotalNrPlayers"]



plt.bar(scalars, height=height, color="turquoise")



plt.title("Total Number of Players per Draft Pick")

plt.xlabel("Draft Pick", fontsize=15)

plt.ylabel("Total Number of Players", fontsize=15)

plt.xticks(list(range(scalars.min(), scalars.max()+1)), fontsize=10)

plt.yticks(fontsize=10)
def subplot(years):



    df = df_allstar.groupby(["Year", "Nationality"])["Player"].count().reset_index().rename(columns={"Player":"TotalNrPlayers"})

    df = df[(df["Year"] >= min(years)) & (df["Year"] <= max(years))]

    

    fig = plt.figure(figsize=(75,30))

    plt.title("Players in the AllStar and their Nationalities({}-{})".format(min(years), max(years)), fontsize=80, pad=40)

    plt.ylabel("AllStar Edition", fontsize=80)

    plt.xlabel("Total Number of Players / Nationality", fontsize=80)

    #years = list(range(2000,2017))

    plt.xticks(years, fontsize=40)

    plt.yticks(list(range(0,30)), fontsize=40)



    ##### Random lambda function for color mapping and other sort of iterative processes requiring random number generation. 

    import random 

    r = lambda: random.randint(0,255)



    #### Mapping Containers used afterwards as helper objects.

    color_map = { nation :'#%02X%02X%02X' % (r(),r(),r()) for nation in df["Nationality"].unique() }

    year_map = { year: i for i, year in enumerate(years) }





    j=0

    reference=np.zeros(len(years))



    for nation in df.sort_values(by="TotalNrPlayers",ascending=True).drop_duplicates(subset=["Nationality"])["Nationality"]:



        total_players = df[df["Nationality"]==nation][["Year", "TotalNrPlayers"]]

        KeepIndices = [ year_map[year] for year in total_players["Year"] ]



        stacked_values = [0 for i in range(0, len(years))]

        values = total_players["TotalNrPlayers"].values



        for i, index in enumerate(KeepIndices):

            stacked_values.insert(index, values[i])

            stacked_values.pop(index+1)



        if j>0:

            plt.bar(years, stacked_values, width=0.4, edgecolor='white', label=nation, bottom=reference, color=color_map[nation], align="center")

            reference += np.array(stacked_values) 

        else: 

            plt.bar(years, stacked_values, width=0.4, edgecolor='white', label=nation, color=color_map[nation], align="center")

            reference = np.array(stacked_values)





        j+=1



    plt.legend(fontsize=40, framealpha=0.85)



plt.tight_layout()

   


period = list(range(2000,2017))



for i in range(0,4):

    init = i*4

    end = init+4

    if init > 10:

        end+=1

        print(init,end)

    subplot(period[init:end])
plt.clf()



plt.rcdefaults()





SMALL_SIZE = 15

MEDIUM_SIZE = 20

BIGGER_SIZE = 25



plt.rc('font', size=SMALL_SIZE)          # controls default text sizes

plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title

plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



df = df_allstar.groupby(["Team", "Selection Type"])["Year"].count().reset_index().rename(columns={"Year":"Total"})



TopTeams = df.groupby(["Team"]).sum()["Total"].sort_values(ascending=False).rename(columns={""}).reset_index()





fig = plt.figure(figsize=(25,15))

plt.title("Proportion of Selection Type per Team for the Period 2000-2016", fontsize=25, pad=50)

plt.ylabel("Totals", fontsize=25)

plt.xlabel("Teams", fontsize=25)

plt.yticks(list(range(0,50)))







##### Random lambda function for color mapping and other sort of iterative processes requiring random number generation. 

import random 

r = lambda: random.randint(0,255)



#### Mapping Containers used afterwards as helper objects.

color_map = { selection_type :'#%02X%02X%02X' % (r(),r(),r()) for selection_type in df["Selection Type"].unique() }





teams = TopTeams["Team"].values

team_map = { team: i for i, team in enumerate(teams) }

reference = np.zeros(len(teams)+1)



selection_types = df["Selection Type"].unique()

df.groupby(["Team", "Selection Type"]).count()





j=0

for selection_type in df.sort_values(by="Total",ascending=True).drop_duplicates(subset="Selection Type")["Selection Type"]:

    

    team_totals = df[df["Selection Type"]==selection_type][["Team", "Total"]]

    KeepIndices = [ team_map[team] for team in team_totals["Team"] ]

    stacked_values = [0 for i in range(0, len(teams))]

    values = team_totals["Total"].values

    

    for i, index in enumerate(KeepIndices):

        stacked_values.insert(index, values[i])

        stacked_values.pop(index+1)



    if j>0:

        plt.bar(teams, stacked_values, width=0.9, edgecolor='white', label=selection_type, bottom=reference, color=color_map[selection_type], align="center")

        reference += np.array(stacked_values) 

    else: 

        plt.bar(teams, stacked_values, width=0.9, edgecolor='white', label=selection_type, color=color_map[selection_type], align="center")

        reference = np.array(stacked_values)

  



    j+=1

    

plt.xticks(teams, rotation=-75)

xlim = plt.xlim()

plt.legend()

    

df = df_allstar.groupby(["Team", "Pos"])["Year"].count().reset_index().rename(columns={"Year":"Total"})



TopTeams = df.groupby(["Team"]).sum()["Total"].sort_values(ascending=False).reset_index()





fig = plt.figure(figsize=(25,15))

plt.title("Proportion of Pos per Team for the Period 2000-2016", fontsize=25).set_position([.5, 1.05])

plt.ylabel("Totals", fontsize=25)

plt.xlabel("Teams", fontsize=25)

plt.yticks(list(range(0,50)))







##### Random lambda function for color mapping and other sort of iterative processes requiring random number generation. 

import random 

r = lambda: random.randint(0,255)



#### Mapping Containers used afterwards as helper objects.

color_map = { pos_type :'#%02X%02X%02X' % (r(),r(),r()) for pos_type in df["Pos"].unique() }





teams = TopTeams["Team"].values

team_map = { team: i for i, team in enumerate(teams) }

reference = np.zeros(len(teams)+1)



pos_types = df["Pos"].unique()

df.groupby(["Team", "Pos"]).count()





j=0

for pos_type in df.sort_values(by="Total",ascending=True).drop_duplicates(subset="Pos")["Pos"]:

    

    team_totals = df[df["Pos"]==pos_type][["Team", "Total"]]

    KeepIndices = [ team_map[team] for team in team_totals["Team"] ]

    stacked_values = [0 for i in range(0, len(teams))]

    values = team_totals["Total"].values

    

    for i, index in enumerate(KeepIndices):

        stacked_values.insert(index, values[i])

        stacked_values.pop(index+1)



    if j>0:

        plt.bar(teams, stacked_values, width=0.9, edgecolor='white', label=pos_type, bottom=reference, color=color_map[pos_type], align="center")

        reference += np.array(stacked_values) 

    else: 

        plt.bar(teams, stacked_values, width=0.9, edgecolor='white', label=pos_type, color=color_map[pos_type], align="center")

        reference = np.array(stacked_values)

  



    j+=1

    

plt.xticks(teams, rotation=-75)

xlim = plt.xlim()

plt.legend()

plt.tight_layout()
title1 = "Number of Participations per Player(2000-2016)"

fig1, ax1 = plt.subplots(figsize=(10,15))





###### Extracting label and scalar data and plotting 

df1 = df_allstar.groupby(["Player"])["Year"].count().reset_index().rename(columns={"Year": "Total(Years)"}).sort_values(by="Total(Years)", ascending=False)

 

n_subsampling = 3

blue_palette = sns.cubehelix_palette(n_colors=len(df1["Total(Years)"][::3]), start=0.2, rot=.7, reverse=True)



ax1.tick_params(pad=30)

plt.title(title1)

ax1.barh(df1["Player"][::3], df1["Total(Years)"][::3], height=0.7, color=blue_palette)



widths = []

patches = list(ax1.patches)

patches.reverse()

for p in patches:



    width = p.get_width()

    if width not in widths:

        height = p.get_y() + 0.2*p.get_height()

        ax1.annotate("{}".format(width), xy=(width, height), xytext=(width+0.1, height))

        widths.append(width)



plt.xticks([])
fig2, ax2 = plt.subplots(figsize=(10,15))





df2 = df_allstar.groupby(["Team"])["Year"].count().reset_index().rename(columns={"Year": "Total(Years)"}).sort_values(by="Total(Years)", ascending=False)

x2, y2 = df2["Total(Years)"], df2["Team"]



purple_palette = sns.cubehelix_palette(n_colors=len(y2), start=2.0, rot=.1, reverse=True)



ax2.tick_params(pad=30)

xticks = list(range(0,x2.max()))

plt.title("Total Number of Participations per Team (2000-2016)")

ax2.barh(y2, x2, height=0.7, color=purple_palette, edgecolor="white")







for p in ax2.patches:



    width = p.get_width()

    height = p.get_y() + 0.4*p.get_height()



    ax2.annotate("{}".format(width), xy=(width, height), xytext=(width+0.1, height))

    



plt.xticks([])

plt.gcf().set_facecolor('white')
