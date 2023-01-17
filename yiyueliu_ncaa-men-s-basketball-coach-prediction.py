import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bq_helper import BigQueryHelper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input director
import os
print(os.listdir("../input"))


bq_assistant = BigQueryHelper(active_project="bigquery-public-data",dataset_name="ncaa_basketball") 
#question: why the dataset_name = "ncaa_basketball"? where can I find "ncaa_basketball"????
bq_assistant.list_tables()
bq_assistant.table_schema("mbb_historical_teams_games")
#"mbb_historical_teams_games"
#win vs loss in each game

bq_assistant.head("mbb_historical_teams_games", num_rows=5)
QUERY = '''SELECT season, market, team_code, points_game,win,opp_market, opp_points_game
            FROM `bigquery-public-data.ncaa_basketball.mbb_historical_teams_games`
            WHERE  2008 <= season AND win = True AND market IS NOT NULL AND opp_market IS NOT NULL 
            ORDER BY market, season, opp_market
'''    
#LIMIT 10
#` is not equal to '!!!!!!!
#bq_assistant.estimate_query_size(QUERY)
df = bq_assistant.query_to_pandas_safe(QUERY)

df.drop_duplicates(inplace=True) #return a new dataset
df.head()
#SELECT
#"mbb_historical_teams_seasons"
#total number of wins and losses of each team in one year(season) from 2008-2016

bq_assistant.head("mbb_historical_teams_seasons", num_rows = 5)
QUERY2 = '''SELECT season, market, wins, losses
            FROM `bigquery-public-data.ncaa_basketball.mbb_historical_teams_seasons`
            WHERE 2008 <= season AND market IS NOT NULL
            ORDER BY market, season
'''
df2 = bq_assistant.query_to_pandas_safe(QUERY2)
df2.drop_duplicates(inplace=True)
df2.head()
#SELECT
#"mbb_teams"
#all school names

bq_assistant.head("mbb_teams", num_rows = 5)
QUERY3 = '''SELECT turner_name, kaggle_team_id, school_ncaa
            FROM `bigquery-public-data.ncaa_basketball.mbb_teams`
            ORDER BY kaggle_team_id
'''

df3 = bq_assistant.query_to_pandas_safe(QUERY3)
df3.drop_duplicates(inplace=True)
df3.head()
#combine df2 and df3
df_points = pd.merge(df2, df3, left_on="market", right_on="turner_name" )
df_points.sort_values(["kaggle_team_id","season"], inplace=True)
df_points.drop(["school_ncaa", "turner_name"], axis=1, inplace=True)

df_points = df_points[["season", "market", "kaggle_team_id", "wins", "losses"]]
df_points.head()
###########################################################
#now doing Google Cloud data
df_coach = pd.read_csv("../input/TeamCoaches.csv")
df_coach = df_coach[["Season", 'TeamID', "CoachName"]]
df_coach.sort_values(["TeamID", "Season"], inplace=True)
df_coach.head()
#final step of preparing data: merger df_coach and df_points
df_final = pd.merge(df_points, df_coach, left_on=["kaggle_team_id", "season"], right_on=["TeamID","Season"])
df_final = df_final[["season", "market", "TeamID", "wins", "losses", "CoachName"]]
df_final.sort_values(["TeamID", "season"], inplace=True)
df_final.head(30)