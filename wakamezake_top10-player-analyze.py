import numpy as np
import pandas as pd
train = pd.read_csv("../input/train_V2.csv")
train.columns
top10_players = train[train["winPlacePerc"] > 0.9]
train["matchType"].unique()
# match-type soloのみ
# fppは「strictly in 1st person」おそらく一人称視点固定
# https://pubg.gamepedia.com/Game_Modes/Solos_FPPより
solo_type = ["solo-fpp", "solo", "normal-solo-fpp", "normal-solo"]
solo_top10_players = top10_players[top10_players["matchType"].isin(solo_type)]
solo_top10_players
skip_columns = ["Id", "groupId", "matchId"]
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
trace1 = go.Histogram(
    x=solo_top10_players["kills"],
    opacity=0.75,
    name = "kills",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
data = [trace1]
layout = go.Layout(barmode='overlay',
                   title='PUBG TOP10-Players(solo) kill histgram',
                   xaxis=dict(title='kills'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# top10playerでも0killがいる
solo_top10_players[solo_top10_players["kills"] == 0]
solo_top10_players[solo_top10_players["kills"] == 0].describe()
total_distance = solo_top10_players["swimDistance"] + solo_top10_players["rideDistance"] + solo_top10_players["walkDistance"]
trace1 = go.Histogram(
    x=total_distance,
    opacity=0.75,
    name = "kills",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
data = [trace1]
layout = go.Layout(barmode='overlay',
                   title='PUBG TOP10-Players(solo) total_distance histgram',
                   xaxis=dict(title='total_distance'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
kill_assists = solo_top10_players["kills"] + solo_top10_players["assists"]
distace_damage_rate = kill_assists / total_distance
distace_damage_rate = distace_damage_rate.replace(np.inf, np.nan).fillna(0)
distace_damage_rate.describe()
distace_total_damage_rate = solo_top10_players["damageDealt"] / total_distance
distace_total_damage_rate = distace_total_damage_rate.replace(np.inf, np.nan).fillna(0)
distace_total_damage_rate.describe()
solo_top10_players["distace_damage_rate"] = distace_damage_rate
solo_top10_players["distace_total_damage_rate"] = distace_total_damage_rate
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(solo_top10_players.corr())
solo_top10_players.columns
