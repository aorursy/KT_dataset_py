# importing libreries and changing their name

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib
# read the excel file
df = pd.read_csv("../input/indian-premier-league-csv-dataset/Ball_by_Ball.csv")
df.head()
# Retrieving the second dataset

df1= pd.read_csv(r'../input/indian-premier-league-csv-dataset/Player.csv')
df2 = df1.drop(["Is_Umpire", "Unnamed: 7"], axis = 1)
# Players = Players[["Player_Id", "Player_Name"]]
df2.head()
# Keeping only the relevant data for analysis

df = df[["Match_Id", "Over_Id", "Striker_Id", "Bowler_Id", "Batsman_Scored", "Extra_Runs"]]
df.head()
# Replacing NaNs with 0. 

df["Extra_Runs"] = pd.to_numeric(df["Extra_Runs"], errors="coerce")
df["Extra_Runs"] = df["Extra_Runs"].fillna(0)

df["Batsman_Scored"] = pd.to_numeric(df["Batsman_Scored"], errors="coerce")
df["Batsman_Scored"] = df["Batsman_Scored"].fillna(0)
df.head(10)