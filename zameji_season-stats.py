import os

os.listdir("../input/stage2wdatafiles")
import numpy as np

import pandas as pd

from sklearn import *



teams = pd.read_csv('../input/stage2wdatafiles/WTeams.csv')

teams2 = pd.read_csv('../input/stage2wdatafiles/WTeamSpellings.csv', encoding='latin-1')



season_dresults = pd.read_csv('../input/stage2wdatafiles/WRegularSeasonDetailedResults.csv')

season_cresults = pd.read_csv('../input/stage2wdatafiles/WRegularSeasonCompactResults.csv')

#tourney_dresults = pd.read_csv('../input/wdatafiles/WNCAATourneyDetailedResults.csv')



season_cresults.describe()
teams2 = teams2.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()

teams2.columns = ['TeamID', 'TeamNameCount']

teams = pd.merge(teams, teams2, how='left', on=['TeamID'])

del teams2
import matplotlib.pyplot as plt



for y in set(season_dresults["Season"].ravel()):

    plt.hist(season_dresults.loc[(season_dresults["Season"]==y),"DayNum"], bins=132)



plt.show()
season_dresults.columns
season_dresults = season_dresults.loc[season_dresults["DayNum"]>100]

season_cresults = season_cresults.loc[season_cresults["DayNum"]>100]



stats = ["TeamID", "Season", "Games", "Wins", "Score","FGM", "FGA", "FGM3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF", "Score_A", "FGM_A", "FGA_A", "FGM3_A", "FTM_A", "FTA_A", "OR_A", "DR_A", "Ast_A", "TO_A", "Stl_A", "Blk_A", "PF_A"]



# Collect stats for each team for each season

frame = np.zeros([len(teams)*len(range(1998,2020)), len(stats)])

season_stats = pd.DataFrame(frame, index=range(len(teams)*len(range(1998,2020))), columns=stats)

season_stats["TeamID"] = [item for sublist in [len(range(1998,2020))*[t_id] for t_id in teams["TeamID"]] for item in sublist]

season_stats["Season"] = len(teams)*[x for x in range(1998,2020)]



for c,r in season_dresults.iterrows():

    

    wstats = r[["W"+x for x in ["TeamID", "FGM", "FGA", "FGM3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]]]

    lstats = r[["L"+x for x in ["TeamID", "FGM", "FGA", "FGM3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]]]

    

    #print(wstats[["WTeamID", "WFGM", "WFGA", "WFGM3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]].values)

    

    season_stats.loc[(season_stats.TeamID == wstats["WTeamID"]) & (season_stats.Season == r["Season"]),

                     ["FGM", "FGA", "FGM3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]] += wstats[["WFGM", "WFGA", "WFGM3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]].values

    season_stats.loc[(season_stats.TeamID == wstats["WTeamID"]) & (season_stats.Season == r["Season"]),

                     ["FGM_A", "FGA_A", "FGM3_A", "FTM_A", "FTA_A", "OR_A", "DR_A", "Ast_A", "TO_A", "Stl_A", "Blk_A", "PF_A"]] += lstats[["LFGM", "LFGA", "LFGM3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF"]].values   

    season_stats.loc[(season_stats.TeamID == lstats["LTeamID"]) & (season_stats.Season == r["Season"]),

                     ["FGM", "FGA", "FGM3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]] += lstats[["LFGM", "LFGA", "LFGM3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF"]].values

    season_stats.loc[(season_stats.TeamID == lstats["LTeamID"]) & (season_stats.Season == r["Season"]),

                     ["FGM_A", "FGA_A", "FGM3_A", "FTM_A", "FTA_A", "OR_A", "DR_A", "Ast_A", "TO_A", "Stl_A", "Blk_A", "PF_A"]] += wstats[["WFGM", "WFGA", "WFGM3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]].values    

    

    season_stats.loc[(season_stats.TeamID == wstats["WTeamID"]) & (season_stats.Season == r["Season"]), ["Games", "Wins"]] += 1   

    season_stats.loc[(season_stats.TeamID == lstats["LTeamID"]) & (season_stats.Season == r["Season"]), "Games"] += 1   



    

for c,r in season_cresults.iterrows():

    

    wstats = r[["W"+x for x in ["TeamID", "Score"]]]

    lstats = r[["L"+x for x in ["TeamID", "Score"]]]

    

    #print(wstats[["WTeamID", "WFGM", "WFGA", "WFGM3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]].values)

    

    season_stats.loc[(season_stats.TeamID == wstats["WTeamID"]) & (season_stats.Season == r["Season"]),

                     "Score"] += wstats["WScore"]

    season_stats.loc[(season_stats.TeamID == wstats["WTeamID"]) & (season_stats.Season == r["Season"]),

                     "Score_A"] += lstats["LScore"]   

    season_stats.loc[(season_stats.TeamID == lstats["LTeamID"]) & (season_stats.Season == r["Season"]),

                     "Score"] += lstats["LScore"]

    season_stats.loc[(season_stats.TeamID == lstats["LTeamID"]) & (season_stats.Season == r["Season"]),

                     "Score_A"] += wstats["WScore"]   

    

    season_stats.loc[(season_stats.TeamID == wstats["WTeamID"]) & (season_stats.Season == r["Season"]), ["Games", "Wins"]] += 1   

    season_stats.loc[(season_stats.TeamID == lstats["LTeamID"]) & (season_stats.Season == r["Season"]), "Games"] += 1   

        

  #  print(season_stats.loc[(season_stats.TeamID == lstats["LTeamID"]) & (season_stats.Season == r["Season"])])

  #  break



season_stats.loc[season_stats["Season"].isin(range(1998,2010))] = season_stats.loc[season_stats["Season"].isin(range(1998,2010))].replace({0:np.nan})

season_stats = season_stats.astype("float64")

season_stats.to_csv("RecentStatsSince1998.csv")
