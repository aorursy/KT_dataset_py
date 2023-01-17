# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/pubg-finish-placement-prediction"))



# Any results you write to the current directory are saved as output.
df_sub = pd.read_csv("../input/dataset/score.csv")
df_test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")
df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()

df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()

df_sub_group = df_sub_group.merge(

    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 

    on="matchId", how="left")

df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)

df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")

df_sub["winPlacePerc"] = df_sub["adjusted_perc"]
df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0

df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1
subset = df_sub.loc[df_sub.maxPlace > 1]

gap = 1.0 / (subset.maxPlace.values - 1)

new_perc = np.around(subset.winPlacePerc.values / gap) * gap

df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc
df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0

assert df_sub["winPlacePerc"].isnull().sum() == 0
df_sub[["Id", "winPlacePerc"]].to_csv("submission.csv", index=False)

print(df_sub['winPlacePerc'].describe())