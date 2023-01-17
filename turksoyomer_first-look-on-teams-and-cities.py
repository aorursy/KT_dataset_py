# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Ignoring future warnings.



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Importing Plotly packages.



import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px
cities = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/Cities.csv")

print(cities.info())

print(cities.head())
wgamecities = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WGameCities.csv")

print(wgamecities.info())

print(wgamecities.head())
wteams = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WTeams.csv")

print(wteams.info())

print(wteams.head())
data = wgamecities

data["WTeamName"] = [wteams.at[wteams[wteams.TeamID == i].index.item(), "TeamName"] for i in data.WTeamID]

data["LTeamName"] = [wteams.at[wteams[wteams.TeamID == i].index.item(), "TeamName"] for i in data.LTeamID]

data["City"] = [cities.at[cities[cities.CityID == i].index.item(), "City"] for i in data.CityID]

data["State"] = [cities.at[cities[cities.CityID == i].index.item(), "State"] for i in data.CityID]

print(data.info())

print(data.head())
teamsWL = []

for team in data.WTeamName.unique():

    wins = len(data[data.WTeamName == team])

    loses = len(data[data.LTeamName == team])

    teamsWL.append([team, wins, loses])

teamsWL = pd.DataFrame(teamsWL, columns=["TeamName", "NumberOfWins", "NumberOfLoses"])

teamsWL["WinningPercentage"] = teamsWL.NumberOfWins / (teamsWL.NumberOfWins+teamsWL.NumberOfLoses)

teamsWL.head()
teams=teamsWL.sort_values(by=["NumberOfWins"], ascending=False)["TeamName"][:10]

wins=teamsWL.sort_values(by=["NumberOfWins"], ascending=False)["NumberOfWins"][:10]

loses=teamsWL.sort_values(by=["NumberOfWins"], ascending=False)["NumberOfLoses"][:10]



fig = go.Figure(data=[

    go.Bar(name='NumberOfWins', x=teams, y=wins, marker_color="#10316b"),

    go.Bar(name='NumberOfLoses', x=teams, y=loses, marker_color="#801336")

])



fig.update_layout(barmode='stack',

                  title="10 Most Winning Teams",

                  xaxis_title="Teams",

                  yaxis_title="Played Games")

fig.show()
teams=teamsWL.sort_values(by=["WinningPercentage"], ascending=False)["TeamName"][:10]

wins=teamsWL.sort_values(by=["WinningPercentage"], ascending=False)["NumberOfWins"][:10]

loses=teamsWL.sort_values(by=["WinningPercentage"], ascending=False)["NumberOfLoses"][:10]

win_lose=teamsWL.sort_values(by=["WinningPercentage"], ascending=False)["WinningPercentage"][:10]



fig = go.Figure(data=[

    go.Bar(name='NumberOfWins', x=teams, y=wins, marker_color="#10316b", text=win_lose),

    go.Bar(name='NumberOfLoses', x=teams, y=loses, marker_color="#801336", text=win_lose)

])



fig.update_layout(barmode='stack',

                  title="10 Most Successful Teams by Winning Percentage",

                  xaxis_title="Teams",

                  yaxis_title="Played Games")

fig.show()
teamsWL2019 = []

for team in data.WTeamName.unique():

    wins = len(data[(data.WTeamName == team) & (data.Season == 2019)])

    loses = len(data[(data.LTeamName == team) & (data.Season == 2019)])

    teamsWL2019.append([team, wins, loses])

teamsWL2019 = pd.DataFrame(teamsWL2019, columns=["TeamName", "NumberOfWins", "NumberOfLoses"])

teamsWL2019["WinningPercentage"] = teamsWL2019.NumberOfWins / (teamsWL2019.NumberOfWins+teamsWL2019.NumberOfLoses)

teamsWL2019.head()
teams=teamsWL2019.sort_values(by=["WinningPercentage"], ascending=False)["TeamName"][:10]

wins=teamsWL2019.sort_values(by=["WinningPercentage"], ascending=False)["NumberOfWins"][:10]

loses=teamsWL2019.sort_values(by=["WinningPercentage"], ascending=False)["NumberOfLoses"][:10]

win_lose=teamsWL2019.sort_values(by=["WinningPercentage"], ascending=False)["WinningPercentage"][:10]



fig = go.Figure(data=[

    go.Bar(name='NumberOfWins', x=teams, y=wins, marker_color="#10316b", text=win_lose),

    go.Bar(name='NumberOfLoses', x=teams, y=loses, marker_color="#801336", text=win_lose)

])



fig.update_layout(barmode='stack',

                  title="10 Most Successful Teams by Winning Percentage at 2019",

                  xaxis_title="Teams",

                  yaxis_title="Played Games")

fig.show()
teams=teamsWL.sort_values(by=["WinningPercentage"])["TeamName"][:10]

wins=teamsWL.sort_values(by=["WinningPercentage"])["NumberOfWins"][:10]

loses=teamsWL.sort_values(by=["WinningPercentage"])["NumberOfLoses"][:10]

win_lose=teamsWL.sort_values(by=["WinningPercentage"])["WinningPercentage"][:10]



fig = go.Figure(data=[

    go.Bar(name='NumberOfLoses', x=teams, y=loses, marker_color="#801336", text=win_lose),

    go.Bar(name='NumberOfWins', x=teams, y=wins, marker_color="#10316b", text=win_lose),

])



fig.update_layout(barmode='stack',

                  title="10 Most Unsuccessful Teams by Winning Percentage",

                  xaxis_title="Teams",

                  yaxis_title="Played Games")

fig.show()
teams=teamsWL2019.sort_values(by=["WinningPercentage"])["TeamName"][:10]

wins=teamsWL2019.sort_values(by=["WinningPercentage"])["NumberOfWins"][:10]

loses=teamsWL2019.sort_values(by=["WinningPercentage"])["NumberOfLoses"][:10]

win_lose=teamsWL2019.sort_values(by=["WinningPercentage"])["WinningPercentage"][:10]



fig = go.Figure(data=[

    go.Bar(name='NumberOfLoses', x=teams, y=loses, marker_color="#801336", text=win_lose),

    go.Bar(name='NumberOfWins', x=teams, y=wins, marker_color="#10316b", text=win_lose),

])



fig.update_layout(barmode='stack',

                  title="10 Most Unsuccessful Teams by Winning Percentage at 2019",

                  xaxis_title="Teams",

                  yaxis_title="Played Games")

fig.show()
teamWinningCities = []

for team in teamsWL.sort_values(by=["WinningPercentage"], ascending=False)["TeamName"][:10]:

    for city in data[data.WTeamName == team].City.unique():

        winNumber = len(data[(data.WTeamName == team) & (data.City == city)])

        loseNumber = len(data[(data.LTeamName == team) & (data.City == city)])

        teamWinningCities.append([team, city, winNumber, loseNumber])

teamWinningCities = pd.DataFrame(teamWinningCities, columns=["TeamName", "City", "NumberOfWins", "NumberOfLoses"])

teamWinningCities["WinningPercentage"] = teamWinningCities.NumberOfWins / (teamWinningCities.NumberOfWins+teamWinningCities.NumberOfLoses)

teamWinningCities.head()
def plotSubPies(team):

    df1 = teamWinningCities[teamWinningCities.TeamName == team].sort_values(by=["TeamName", "NumberOfWins", "WinningPercentage"], ascending=False)

    label1 = df1.City[:5]

    value1 = df1.NumberOfWins[:5]

    color1 = ["#052F5F", "#003777", "#0650A5", "#0774F2", "#7AA5D6"]

    

    df2 = teamWinningCities[teamWinningCities.TeamName == team].sort_values(by=["TeamName", "NumberOfLoses", "WinningPercentage"], ascending=False)

    label2 = df2.City[:5]

    value2 = df2.NumberOfLoses[:5]

    color2 =["#580C1F", "#74121D", "#A7333F", "#A51C30", "#C52233"]

    

    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],

                        subplot_titles=[team+"'s Most Winning Cities", team+"'s Most Losing Cities"])

    fig.add_trace(go.Pie(labels=label1, values=value1, name=team+"'s Most Winning Cities", marker_colors=color1),

                  1, 1)

    fig.add_trace(go.Pie(labels=label2, values=value2, name=team+"'s Most Losing Cities", marker_colors=color2),

                  1, 2)



    fig.update_traces(hoverinfo="label+percent")



    fig.update_layout(

        title_text=team+"'s Statistics for Cities")

    fig.show()
print("Top 10 Teams:")

for i, team in enumerate(teamsWL.sort_values(by=["WinningPercentage"], ascending=False)["TeamName"][:10]):

    print("{}. {}".format(i+1, team))
plotSubPies("Connecticut")
plotSubPies("Baylor")
plotSubPies("Notre Dame")
plotSubPies("Stanford")
plotSubPies("WI Green Bay")
plotSubPies("FL Gulf Coast")
plotSubPies("Maryland")
plotSubPies("Gonzaga")
plotSubPies("Princeton")
plotSubPies("James Madison")