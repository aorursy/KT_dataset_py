import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style="whitegrid")
df_deliveries = pd.read_csv("../input/deliveries.csv")

#df_matches = pd.read_csv("../input/matches.csv")



df_deliveries["bowl"] = 1

df_deliveries.head(2)
df_teams = df_deliveries.groupby(["batting_team"]).sum().reset_index()

df_teams = df_teams[["batting_team", "wide_runs", "bye_runs", "legbye_runs", "noball_runs", "total_runs", "batsman_runs", "bowl"]]

df_teams.head(2)
plot_columns = ["batting_team", "total_runs", "batsman_runs", "bowl"]

df_runs = df_teams[plot_columns].sort_values(by=["total_runs"], ascending=[False])



f, ax = plt.subplots(figsize=(6, 6))

df_runs.plot.barh(ax=ax)

f.suptitle("Team's Runs and Bowls", fontsize=14)

wrap = ax.set_yticklabels(list(df_runs["batting_team"]))
plot_columns = ["batting_team", "wide_runs", "bye_runs", "legbye_runs", "noball_runs"]

df_extras = df_teams[plot_columns].sort_values(by=["wide_runs"], ascending=[False])



f, ax = plt.subplots(figsize=(6, 6))

df_extras.plot.barh(ax=ax)

f.suptitle("Team's Extra Runs", fontsize=14)

wrap = ax.set_yticklabels(list(df_extras["batting_team"]))
df_batsman = df_deliveries.groupby(["batsman"]).sum(

).reset_index().nlargest(10, 'batsman_runs')



df_batsman = df_batsman[["batsman", "batsman_runs", "bowl"]]

df_batsman["Strike Rate"] = df_batsman.apply(

    lambda row: int((row["batsman_runs"] * 100) / row["bowl"]), axis=1)



df_batsman.head(2)
df_batsman = df_batsman.sort_values(by=["batsman_runs"], ascending=[True])



f, ax = plt.subplots(figsize=(8, 4))

df_batsman[["batsman", "batsman_runs", "bowl"]].plot.barh(ax=ax)

f.suptitle("Top 10 Batsman (as per Runs scored)", fontsize=14)

wrap = ax.set_yticklabels(list(df_batsman["batsman"]))



rects = ax.patches

bar_labels = list(df_batsman["Strike Rate"])



for i in range(len(bar_labels)):

  label = "Strike Rate: " + str(bar_labels[i])

  ax.text(1500, rects[i].get_y(), label, ha='center',

          va='bottom', size='smaller', color="white")
df_batsmanInTeam = df_deliveries.groupby(

    ["batsman", "batting_team"]).sum().reset_index().nlargest(10, 'batsman_runs')

    

df_batsmanInTeam = df_batsmanInTeam[

    ["batsman", "batting_team", "batsman_runs", "bowl"]]

    

df_batsmanInTeam["Strike Rate"] = df_batsmanInTeam.apply(

    lambda row: int((row["batsman_runs"] * 100) / row["bowl"]), axis=1)

    

df_batsmanInTeam.head(2)
df_batsmanInTeam = df_batsmanInTeam.sort_values(

    by=["batsman_runs"], ascending=[True])



f, ax = plt.subplots(figsize=(8, 4))

df_batsmanInTeam[["batsman", "batsman_runs", "bowl"]].plot.barh(ax=ax)

f.suptitle("Top 10 Batsman by Team (as per Runs scored)", fontsize=14)



team_initials = ['.'.join(name[0].upper() for name in team_name.split())

                 for team_name in list(df_batsmanInTeam["batting_team"])]

                 

player_team_y_label = [player + " (" + team + ")" for player,

                       team in zip(list(df_batsmanInTeam["batsman"]), team_initials)]



wrap = ax.set_yticklabels(player_team_y_label)



rects = ax.patches

bar_labels = list(df_batsmanInTeam["Strike Rate"])



for i in range(len(bar_labels)):

  label = "Strike Rate: " + str(bar_labels[i])

  ax.text(1000, rects[i].get_y(), label, ha='center',

          va='bottom', size='smaller', color="white")
df_batsmanRunCount = df_deliveries.drop(

    df_deliveries[df_deliveries["batsman_runs"] == 0].index).reset_index()

    

df_batsmanRunCount = df_batsmanRunCount.groupby(

    ["batsman", "total_runs"]).sum().reset_index()

    

df_batsmanRunCount = df_batsmanRunCount.rename(

    columns={'bowl': 'count', 'total_runs': 'Run Type'})

    

df_batsmanRunCount = df_batsmanRunCount[["batsman", "Run Type", "count"]]

df_batsmanRunCount.head(2)
f, ax = plt.subplots(2, 2, figsize=(9, 6))

f.subplots_adjust(top=1)

ax_x = 0

ax_y = 0

color_counter = 0  # for color in bars

colors = ["#999966", "#8585ad", "#c4ff4d", "#ffad33"]



for run_type in [1, 2, 4, 6]:

  df_batsmanRunCount_run_type = df_batsmanRunCount[df_batsmanRunCount[

      "Run Type"] == run_type].nlargest(10, 'count').sort_values(by=["count"], ascending=[True])

      

  df_batsmanRunCount_run_type[["batsman", "count"]].plot.barh(

      ax=ax[ax_x, ax_y], legend=False, color=colors[color_counter])

      

  ax[ax_x, ax_y].set_yticklabels(

      list(df_batsmanRunCount_run_type["batsman"]), fontsize=6)

      

  ax[ax_x, ax_y].title.set_text("Most " + str(run_type) + "'s by Batsman")



  color_counter = color_counter + 1



  # for proper subplots

  if ax_y == 1:

    ax_y = 0

    ax_x = ax_x + 1

  else:

    ax_y = ax_y + 1
df_dismissed = df_deliveries.groupby(["dismissal_kind"]).sum(

).reset_index().nlargest(5, 'bowl').reset_index()



df_dismissed = df_dismissed[["dismissal_kind", "bowl"]]

df_dismissed = df_dismissed.rename(columns={'bowl': 'count'})

df_dismissed.head(2)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

f.suptitle("Top 5 Dismissal Kind", fontsize=14)



df_dismissed.plot.bar(ax=ax1, legend=False)

ax1.set_xticklabels(list(df_dismissed["dismissal_kind"]), fontsize=8)



for tick in ax1.get_xticklabels():

  tick.set_rotation(0)



df_dismissed["count"].plot.pie(ax=ax2, labels=df_dismissed[

                               "dismissal_kind"], autopct='%1.1f%%', fontsize=8)

wrap = ax2.set_ylabel('')
df_dismissedBatsman = df_deliveries.groupby(

    ["dismissal_kind", "batsman"]).sum().reset_index()

    

df_dismissedBatsman = df_dismissedBatsman[

    ["batsman", "dismissal_kind", "bowl"]]

    

df_dismissedBatsman = df_dismissedBatsman.rename(columns={'bowl': 'count'})

df_dismissedBatsman.head(2)
f, ax = plt.subplots(2, 2, figsize=(9, 6))

f.subplots_adjust(top=1)

ax_x = 0

ax_y = 0

color_counter = 0  # for bars color 

colors = ["#999966", "#8585ad", "#c4ff4d", "#ffad33"]

common_dismissal_kind = ["bowled", "caught",

                         "run out", "lbw"]  # as seen in above plot



for dismissal_kind in common_dismissal_kind:

  df_dismissedBatsmanKind = df_dismissedBatsman[df_dismissedBatsman[

      "dismissal_kind"] == dismissal_kind].nlargest(10, 'count').sort_values(by=["count"], ascending=[True])

      

  df_dismissedBatsmanKind[["batsman", "count"]].plot.barh(

      ax=ax[ax_x, ax_y], legend=False, color=colors[color_counter])

      

  ax[ax_x, ax_y].set_yticklabels(

      list(df_dismissedBatsmanKind["batsman"]), fontsize=6)

      

  ax[ax_x, ax_y].title.set_text("Batsmen dismissed by: " + str(dismissal_kind))



  color_counter = color_counter + 1



  # for proper subplots

  if ax_y == 1:

    ax_y = 0

    ax_x = ax_x + 1

  else:

    ax_y = ax_y + 1
df_bowlersOvers = df_deliveries.groupby(

    ["bowler", "over", "match_id"]).sum().reset_index()

    

df_bowlersOvers["total_overs"] = 1

df_bowlersOvers = df_bowlersOvers[["bowler", "total_overs", "bowl", "total_runs",

                                   "wide_runs", "bye_runs", "legbye_runs", "noball_runs", "extra_runs"]]

                                   

df_bowlersOvers = df_bowlersOvers.groupby(["bowler"]).sum().reset_index()

df_bowlersOvers.head(2)
df_bowlersOversMax = df_bowlersOvers.nlargest(

    10, 'total_overs').sort_values(by=["total_overs"], ascending=[True])

    

df_bowlersOversMax["Economy"] = df_bowlersOversMax.apply(

    lambda row: "{:.2f}".format(row["total_runs"] / row["total_overs"]), axis=1)



f, ax = plt.subplots(figsize=(8, 4))

df_bowlersOversMax[["bowler", "total_overs"]].plot.barh(ax=ax)

f.suptitle("Top 10 Bowlers (as per overs bowled)", fontsize=14)

wrap = ax.set_yticklabels(list(df_bowlersOversMax["bowler"]))



rects = ax.patches

bar_labels = list(df_bowlersOversMax["Economy"])



for i in range(len(bar_labels)):

  label = "Economy: " + str(bar_labels[i])

  ax.text(200, rects[i].get_y(), label, ha='center',

          va='bottom', size='smaller', color="w")
f, ax = plt.subplots(figsize=(8, 4))

df_bowlersExtra = df_bowlersOvers.nlargest(10, 'extra_runs')

df_bowlersExtra[["bowler", "wide_runs", "bye_runs",

                 "legbye_runs", "noball_runs"]].plot.barh(ax=ax)

                 

f.suptitle("Top 10 Bowlers (as per extra runs)", fontsize=14)

wrap = ax.set_yticklabels(list(df_bowlersExtra["bowler"]))
# obstructing the field, retired hurt, run out does not account for bowler's wicket

bowlerWicketsType = ['bowled', 'caught',

                     'caught and bowled', 'hit wicket', 'lbw', 'stumped']

                     

df_bowlerWickets = df_deliveries[df_deliveries[

    'dismissal_kind'].isin(bowlerWicketsType)]

    

df_bowlerWickets = df_bowlerWickets.rename(columns={'bowl': 'Wickets'})



df_bowlerWickets = df_bowlerWickets.groupby(["bowler"]).sum(

).reset_index().nlargest(10, 'Wickets').reset_index()



df_bowlerWickets = df_bowlerWickets[["bowler", "Wickets"]]

df_bowlerWickets.head(2)
df_bowlerWickets = df_bowlerWickets.sort_values(by=["Wickets"], ascending=[True])



f, ax = plt.subplots(figsize=(8, 4))

df_bowlerWickets.plot.barh(ax=ax)

f.suptitle("Top 10 Bowlers (as per extra runs)", fontsize=14)

wrap = ax.set_yticklabels(list(df_bowlerWickets["bowler"]))
df_bowlerWickets_dismissal_kind = df_deliveries.groupby(

    ["dismissal_kind", "bowler"]).sum().reset_index()

    

df_bowlerWickets_dismissal_kind = df_bowlerWickets_dismissal_kind[

    ["dismissal_kind", "bowler", "bowl"]]

    

df_bowlerWickets_dismissal_kind = df_bowlerWickets_dismissal_kind.rename(

    columns={'bowl': 'Wickets'})

    

# obstructing the field, retired hurt, run out does not account for bowler's wicket

bowlerWicketsType = ['bowled', 'caught',

                     'caught and bowled', 'hit wicket', 'lbw', 'stumped']

                     

df_bowlerWickets_dismissal_kind = df_bowlerWickets_dismissal_kind[

    df_bowlerWickets_dismissal_kind['dismissal_kind'].isin(bowlerWicketsType)]

    

df_bowlerWickets_dismissal_kind.head(2)
f, ax = plt.subplots(3, 2, figsize=(9, 10))

f.subplots_adjust(top=1)

ax_x = 0

ax_y = 0

color_counter = 0  # for color in bars

colors = ["#D00000", "#8585ad", "#c4ff4d", "#ffad33", "#5f646d", "#999966"]



for dismissal_kind in bowlerWicketsType:

  df_dismissedBowlerKind = df_bowlerWickets_dismissal_kind[df_bowlerWickets_dismissal_kind[

      "dismissal_kind"] == dismissal_kind].nlargest(10, 'Wickets').sort_values(by=["Wickets"], ascending=[True])

      

  df_dismissedBowlerKind[["bowler", "Wickets"]].plot.barh(

      ax=ax[ax_x, ax_y], legend=False, color=colors[color_counter])

      

  ax[ax_x, ax_y].set_yticklabels(

      list(df_dismissedBowlerKind["bowler"]), fontsize=6)

      

  ax[ax_x, ax_y].title.set_text("Top 10: " + str(dismissal_kind))



  color_counter = color_counter + 1



  # for proper subplots

  if ax_y == 1:

    ax_y = 0

    ax_x = ax_x + 1

  else:

    ax_y = ax_y + 1