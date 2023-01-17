import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print(os.listdir('../input'))
team_info = pd.read_csv('../input/team_info.csv', delimiter=',', usecols=['team_id', 'shortName', 'abbreviation'])

team_info.head()
game_teams_stats = pd.read_csv('../input/game_teams_stats.csv', delimiter=',', usecols=['game_id', 'team_id', 'HoA', 'won', 'settled_in', 'goals', 'shots', 'powerPlayGoals'])
game_teams_stats.head()
b = game_teams_stats.goals.value_counts().to_frame()

game_teams_stats.goals.value_counts().to_frame().plot.bar()

b = pd.DataFrame([{"goals": int(b.loc[[0, 1, 2, 3] , :].sum())}, {"goals": int(b.loc[[4,5,6,7,8,9,10], :].sum())}], index = ["3 or less goal games", "4 or more goal games"]).plot.bar()

b.set_ylabel("Games")

b.get_legend().remove()

plt.xticks([0,1], ["3 or less goal games", "4 or more goal games"], rotation="horizontal")
game_teams_stats.loc[(game_teams_stats["won"] == True) & (game_teams_stats["settled_in"] != "SO")]["goals"].value_counts().plot.bar()
print(game_teams_stats.loc[(game_teams_stats["won"] == True)]["team_id"].value_counts().mean())

print(game_teams_stats.loc[(game_teams_stats["won"] == True) & (game_teams_stats["team_id"] == 14)]["team_id"].value_counts())

print(game_teams_stats.loc[(game_teams_stats["won"] == True) & (game_teams_stats["team_id"] == 21)]["team_id"].value_counts())

print(game_teams_stats.loc[(game_teams_stats["won"] == True) & (game_teams_stats["team_id"] == 26)]["team_id"].value_counts())

print(game_teams_stats.loc[(game_teams_stats["won"] == True) & (game_teams_stats["team_id"] == 2)]["team_id"].value_counts())

print(game_teams_stats.loc[(game_teams_stats["won"] == True) & (game_teams_stats["team_id"] == 22)]["team_id"].value_counts())
game_teams_stats_home = game_teams_stats[game_teams_stats.HoA == 'home']

game_teams_stats_away = game_teams_stats[game_teams_stats.HoA == 'away']

game_teams_combined_result = game_teams_stats_home.merge(game_teams_stats_away, left_on="game_id", right_on="game_id", suffixes=("_home", "_away")).drop(

    ["HoA_home", "HoA_away", "won_away", "settled_in_away"], axis=1

).rename(columns={"settled_in_home": "settled_in"}).sort_values(by = ["game_id"])

game_teams_combined_result.head()
name_replace_dict = {"team_id_away": "team_id", "team_id_home": "team_id"}

game_losing_team = game_teams_combined_result.loc[(game_teams_combined_result["won_home"] == True)][["game_id", "team_id_away", "settled_in"]].rename(columns = name_replace_dict).append(

    game_teams_combined_result.loc[(game_teams_combined_result["won_home"] == False)][["game_id", "team_id_home", "settled_in"]].rename(columns = name_replace_dict)

).sort_values(by = ["game_id"])

game_losing_team.head()
game_plays = pd.read_csv('../input/game_plays.csv', delimiter=',', usecols=['play_id', 'game_id', 'play_num', 'team_id_for', 'team_id_against', 'event', 'secondaryType', 'period', 'periodType'])
game_plays_goals = game_plays.loc[(game_plays["event"] == 'Goal') & (game_plays["periodType"] != 'SHOOTOUT')].sort_values(by = ["game_id", "play_num"])

game_plays_goals.head()
lead_data = {}

lost_lead_game_data = {}
for index, row in game_plays_goals.iterrows():

    

    game_id = row["game_id"]

    team_id_for = row["team_id_for"]

    team_id_against = row["team_id_against"]

    

    if game_id not in lead_data:

        lead_data[game_id] = {}

        lead_data[game_id]["largest_lead"] = 0

        lead_data[game_id][team_id_for] = 0

        lead_data[game_id][team_id_against] = 0        



    lead_data[game_id][team_id_for] += 1

    

    score_dif = lead_data[game_id][team_id_for] - lead_data[game_id][team_id_against]

    

       

    if score_dif >= 2:

        # >= because wanna know latest lead

        if score_dif >= lead_data[game_id]["largest_lead"]:

            if game_id in lost_lead_game_data:

                if lost_lead_game_data[game_id]["largest_lead_team"] == team_id_for:

                    lead_data[game_id]["largest_lead"] = score_dif

                    lead_data[game_id]["largest_lead_score"] = str(lead_data[game_id][team_id_for]) + "-" + str(lead_data[game_id][team_id_against])

                    lead_data[game_id]["largest_lead_team"] = team_id_for

            else:

                lead_data[game_id]["largest_lead"] = score_dif

                lead_data[game_id]["largest_lead_score"] = str(lead_data[game_id][team_id_for]) + "-" + str(lead_data[game_id][team_id_against])

                lead_data[game_id]["largest_lead_team"] = team_id_for



            if game_losing_team.loc[game_losing_team["game_id"] == game_id]["team_id"].squeeze() == team_id_for:

                lead_data[game_id]["winning_team"] = team_id_against

                lost_lead_game_data[game_id] = lead_data[game_id]

                lost_lead_game_data[game_id]["settled_in"] = game_losing_team.loc[game_losing_team["game_id"] == game_id]["settled_in"].squeeze()



    elif score_dif == 0 and game_id in lost_lead_game_data:

        if "period_tied" not in lost_lead_game_data[game_id]:

            lost_lead_game_data[game_id]["period_tied"] = row["period"]
print(len(lead_data))

print("###")

print(len(lost_lead_game_data))

game_plays.loc[(game_plays["event"] == 'Goal') & (game_plays["game_id"] == 2010020007)]
dict_to_pd_prep = {}

final_pd_columns = ["game_id", "largest_lead_score", "largest_lead", "losing_team", "winning_team", "period_tied", "settled_in"]
count = 0

for game_id, data in lost_lead_game_data.items():

    if team_info.loc[team_info["team_id"] == data["largest_lead_team"]]["abbreviation"].squeeze() != "VGK" and team_info.loc[team_info["team_id"] == data["winning_team"]]["abbreviation"].squeeze() != "VGK":

        dict_to_pd_prep[count] = []

        dict_to_pd_prep[count].append(game_id)

        dict_to_pd_prep[count].append(data["largest_lead_score"])

        dict_to_pd_prep[count].append(data["largest_lead"])

        

        # Thrashers and Phoenix relocated to Winnipeg and Arizona

        if team_info.loc[team_info["team_id"] == data["largest_lead_team"]]["abbreviation"].squeeze() == "PHX":

            dict_to_pd_prep[count].append("ARI")

        elif team_info.loc[team_info["team_id"] == data["largest_lead_team"]]["abbreviation"].squeeze() == "ATL":

            dict_to_pd_prep[count].append("WPG")

        else:

            dict_to_pd_prep[count].append(team_info.loc[team_info["team_id"] == data["largest_lead_team"]]["abbreviation"].squeeze())

        

        if team_info.loc[team_info["team_id"] == data["winning_team"]]["abbreviation"].squeeze() == "PHX":

            dict_to_pd_prep[count].append("ARI")

        elif team_info.loc[team_info["team_id"] == data["winning_team"]]["abbreviation"].squeeze() == "ATL":

            dict_to_pd_prep[count].append("WPG")

        else:

            dict_to_pd_prep[count].append(team_info.loc[team_info["team_id"] == data["winning_team"]]["abbreviation"].squeeze())



            dict_to_pd_prep[count].append(data["period_tied"])

        dict_to_pd_prep[count].append(data["settled_in"])



        count += 1
lead_losers_pd = pd.DataFrame.from_dict(dict_to_pd_prep, orient='index', columns=final_pd_columns)
lead_losers_pd.head(4)
all_score_plot = lead_losers_pd.largest_lead_score.value_counts().to_frame().plot.bar()

all_score_plot.set_title("Distribution of lost leads", fontdict = {"fontsize": 18})

all_score_plot.set_xlabel("Scores")

all_score_plot.set_ylabel("Total")

all_score_plot.get_legend().remove()
game_losing_team["settled_in"].value_counts().to_frame()
all_game_settled_in_plot = game_losing_team["settled_in"].value_counts().plot.bar(color=["green", "blue", "red"])

all_game_settled_in_plot.set_title("Settled in (All games)", fontdict = {"fontsize": 18})

all_game_settled_in_plot.set_xlabel("Settled in")

all_game_settled_in_plot.set_ylabel("Total")
lead_losers_pd["settled_in"].value_counts().to_frame()
gt_2_game_settled_in_plot = lead_losers_pd["settled_in"].value_counts().plot.bar(color=["green", "blue", "red"])

gt_2_game_settled_in_plot.set_title("Settled in (Lost leads >= 2)", fontdict = {"fontsize": 18})

gt_2_game_settled_in_plot.set_xlabel("Settled in")

gt_2_game_settled_in_plot.set_ylabel("Total")
lead_losers_pd.loc[lead_losers_pd["largest_lead"] >= 3]["settled_in"].value_counts().to_frame()
gt_3_game_settled_in_plot = lead_losers_pd.loc[lead_losers_pd["largest_lead"] >= 3]["settled_in"].value_counts().plot.bar(color=["green", "blue", "red"])

gt_3_game_settled_in_plot.set_title("Settled in (Lost leads >= 3)", fontdict = {"fontsize": 18})

gt_3_game_settled_in_plot.set_xlabel("Settled in")

gt_3_game_settled_in_plot.set_ylabel("Total")

gt_3_game_settled_in_plot.set_xticklabels(['REG', 'OT', "SO"]) 
gt_4_game_settled_in_plot = lead_losers_pd.loc[lead_losers_pd["largest_lead"] >= 4]["settled_in"].value_counts().plot.bar(color=["blue", "green", "red"])

gt_4_game_settled_in_plot.set_title("Settled in (Lost leads >= 4)", fontdict = {"fontsize": 18})

gt_4_game_settled_in_plot.set_xlabel("Settled in")

gt_4_game_settled_in_plot.set_ylabel("Total")
losing_team_counts = lead_losers_pd["losing_team"].value_counts().plot.bar()

losing_team_counts.set_title("Lost Leads per Team", fontdict = {"fontsize": 18})

losing_team_counts.set_xlabel("Team")

losing_team_counts.set_ylabel("Total")
comeback_team_counts = lead_losers_pd["winning_team"].value_counts().plot.bar()

comeback_team_counts.set_title("Comeback wins per Team", fontdict = {"fontsize": 18})

comeback_team_counts.set_xlabel("Team")

comeback_team_counts.set_ylabel("Total")
nyi_leads_plot = lead_losers_pd.loc[lead_losers_pd["losing_team"] == "NYI"]["largest_lead_score"].value_counts().plot.bar(color="#F47E2D")

nyi_leads_plot.set_title("Distribution of lost leads (NYI)", fontdict = {"fontsize": 18})

nyi_leads_plot.set_xlabel("Scores")

nyi_leads_plot.set_ylabel("Total")

nyi_leads_plot.set_ylim(top=30)
tor_leads_plot = lead_losers_pd.loc[lead_losers_pd["losing_team"] == "TOR"]["largest_lead_score"].value_counts().plot.bar(color="#003876")

tor_leads_plot.set_title("Distribution of lost leads (TOR)", fontdict = {"fontsize": 18})

tor_leads_plot.set_xlabel("Scores")

tor_leads_plot.set_ylabel("Total")

tor_leads_plot.set_ylim(top=30)
pit_leads_plot = lead_losers_pd.loc[lead_losers_pd["winning_team"] == "PIT"]["largest_lead_score"].value_counts().plot.bar(color="#FFC80C")

pit_leads_plot.set_title("Distribution of comeback wins (PIT)", fontdict = {"fontsize": 18})

pit_leads_plot.set_xlabel("Scores")

pit_leads_plot.set_ylabel("Total")

pit_leads_plot.set_ylim(top=30)
pit_leads_plot = lead_losers_pd.loc[lead_losers_pd["winning_team"] == "MTL"]["largest_lead_score"].value_counts().plot.bar(color="red")

pit_leads_plot.set_title("Distribution of comeback wins (MTL)", fontdict = {"fontsize": 18})

pit_leads_plot.set_xlabel("Scores")

pit_leads_plot.set_ylabel("Total")

pit_leads_plot.set_ylim(top=30)
lead_losers_pd["largest_lead_score"].value_counts().multiply(1/len(lead_losers_pd["winning_team"].value_counts())).to_frame()
team_names = lead_losers_pd["winning_team"].unique()
dicty = {}

for team in team_names:

    result = lead_losers_pd["largest_lead_score"].value_counts().multiply(1/len(team_names)).subtract(lead_losers_pd.loc[lead_losers_pd["losing_team"] == team]["largest_lead_score"].value_counts())

    dicty[team] = result.divide(lead_losers_pd["largest_lead_score"].value_counts().multiply(1/len(team_names)))
nyi = dicty["NYI"].plot.bar(color="#F47E2D")

nyi.set_ylim((-1, 1))

nyi.set_title("Lost Leads Difference with League Average (NYI)", fontdict = {"fontsize": 18})

nyi.set_xlabel("Scores")

nyi.set_ylabel("Percentage off Average")
tor = dicty["TOR"].plot.bar(color="#003876")

tor.set_ylim((-1, 1))

tor.set_title("Lost Leads Difference with League Average (TOR)")

tor.set_xlabel("Scores")

tor.set_ylabel("Percentage off Average")
pit = dicty["TBL"].plot.bar(color="#003E7E")

pit.set_ylim((-1, 1))

pit.set_title("Lost Leads Difference with League Average (TBL)")

pit.set_xlabel("Scores")

pit.set_ylabel("Percentage off Average")
pit = dicty["COL"].plot.bar(color="#003E7E")

pit.set_ylim((-1, 1))

pit.set_title("Lost Leads Difference with League Average (COL)")

pit.set_xlabel("Scores")

pit.set_ylabel("Percentage off Average")
count = 0

fig = plt.figure(figsize=(120, 120))

for step in range(30):

    ax = fig.add_subplot(5, 6, count+1)

#     temp_data_frame = df[df['step'].str.contains(step)]

    dicty[team_names[count]].plot(x = 'Score', y = 'Percentage off Average', kind='bar', legend=True, fontsize=40, ax=ax).set_ylim((-1, 1))

    ax.set_title(team_names[count], fontdict = {"fontsize": 60})

    ax.set_xlabel("Score", fontsize=60)

    ax.set_ylabel("Percentage off Average", fontsize=60)

    ax.get_legend().remove()

    count += 1