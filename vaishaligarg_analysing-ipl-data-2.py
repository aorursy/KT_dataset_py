import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





matches = pd.read_csv('../input/matches.csv')

matches["type"] = "pre-qualifier"

for year in range(2008, 2017):

   final_match_index = matches[matches['season']==year][-1:].index.values[0]

   matches = matches.set_value(final_match_index, "type", "final")

   matches = matches.set_value(final_match_index-1, "type", "qualifier-2")

   matches = matches.set_value(final_match_index-2, "type", "eliminator")

   matches = matches.set_value(final_match_index-3, "type", "qualifier-1")



matches.groupby(["type"])["id"].count()

deliveries = pd.read_csv('../input/deliveries.csv')
batsman_grp = deliveries.groupby(["match_id", "inning", "batting_team", "batsman"])

batsmen = batsman_grp["batsman_runs"].sum().reset_index()



# Ignore the wide balls.

balls_faced = deliveries[deliveries["wide_runs"] == 0]

balls_faced = balls_faced.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()

balls_faced.columns = ["match_id", "inning", "batsman", "balls_faced"]

batsmen = batsmen.merge(balls_faced, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")



fours = deliveries[ deliveries["batsman_runs"] == 4]

sixes = deliveries[ deliveries["batsman_runs"] == 6]



fours_per_batsman = fours.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()

sixes_per_batsman = sixes.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()



fours_per_batsman.columns = ["match_id", "inning", "batsman", "4s"]

sixes_per_batsman.columns = ["match_id", "inning", "batsman", "6s"]



batsmen = batsmen.merge(fours_per_batsman, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")

batsmen = batsmen.merge(sixes_per_batsman, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")

batsmen['SR'] = np.round(batsmen['batsman_runs'] / batsmen['balls_faced'] * 100, 2)



for col in ["batsman_runs", "4s", "6s", "balls_faced", "SR"]:

    batsmen[col] = batsmen[col].fillna(0)



dismissals = deliveries[ pd.notnull(deliveries["player_dismissed"])]

dismissals = dismissals[["match_id", "inning", "player_dismissed", "dismissal_kind", "fielder"]]

dismissals.rename(columns={"player_dismissed": "batsman"}, inplace=True)

batsmen = batsmen.merge(dismissals, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")



batsmen = matches[['id','season']].merge(batsmen, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)

batsmen.head()
batsman_runsperseason = batsmen.groupby(['season', 'batting_team', 'batsman'])['batsman_runs'].sum().reset_index()

batsman_runsperseason = batsman_runsperseason.groupby(['season', 'batsman'])['batsman_runs'].sum().unstack().T

batsman_runsperseason['Total'] = batsman_runsperseason.sum(axis=1) #add total column to find batsman with the highest runs

batsman_runsperseason = batsman_runsperseason.sort_values(by = 'Total', ascending = False).drop('Total', 1)

batsman_runsperseason.head()
ax = batsman_runsperseason[:5].T.plot()
batsman_runs = batsmen.groupby(['batsman'])['batsman_runs', '4s', '6s'].sum().reset_index()

batsman_runs['4s_6s'] = batsman_runs['4s'] * 4 + batsman_runs['6s'] * 6

batsman_runs['pct_boundaries'] = np.round(batsman_runs['4s_6s'] / batsman_runs['batsman_runs'] * 100, 2)

batsman_runs = batsman_runs.sort_values(by = 'batsman_runs', ascending = False)

batsman_runs[:10].plot(x= 'batsman', y = 'pct_boundaries', kind = 'barh')
batsmen_dismissal = batsmen['dismissal_kind'].value_counts()

fig, ax = plt.subplots()

ax.pie(batsmen_dismissal, labels = None, autopct='%1.1f%%', startangle=90, shadow = True)

ax.legend(loc='best', labels=batsmen_dismissal.index)
batsmen_SR = batsmen.groupby(['season','batsman'])['batsman_runs', 'balls_faced'].sum().reset_index()

batsmen_SR['SR'] = np.round(batsmen_SR['batsman_runs'] / batsmen_SR['balls_faced'] * 100, 2)

batsmen_SR.groupby(['batsman', 'season']).sum().sort_values(by = 'batsman_runs', ascending = False)[:5]