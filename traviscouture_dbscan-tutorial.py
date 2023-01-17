import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.cluster import DBSCAN

        

finch_data_2012 = "../input/geospiza-scandens-beak-evolution/finch_beaks_2012.csv"

finch_data_1975 = "../input/geospiza-scandens-beak-evolution/finch_beaks_1975.csv"



finch_2012_df = pd.read_csv(filepath_or_buffer=finch_data_2012)

finch_1975_df = pd.read_csv(filepath_or_buffer=finch_data_1975)

finch_1975_df.columns = finch_2012_df.columns
sns.relplot(x="blength", y="bdepth", data=finch_1975_df,)

sns.relplot(x="blength", y="bdepth", data=finch_2012_df)
# Lets prepare out data for clustering by pairing out blength and bdepth columns into a single column repsrsenting a point in 2D space.

finch_2012_beak_points = list(zip(finch_2012_df["blength"], finch_2012_df["bdepth"]))

finch_1975_beak_points = list(zip(finch_1975_df["blength"], finch_1975_df["bdepth"]))

finch_1975_df["beak_points"] = finch_1975_beak_points

finch_2012_df["beak_points"] = finch_2012_beak_points



# Cluster time!

finch_1975_df["cluster"] = DBSCAN().fit(finch_1975_beak_points).labels_

finch_2012_df["cluster"] = DBSCAN().fit(finch_2012_beak_points).labels_

print(finch_1975_df.head())

print()

print(finch_2012_df.head())
sns.relplot(x="blength", y="bdepth", hue="cluster", data=finch_1975_df)

sns.relplot(x="blength", y="bdepth", hue="cluster", data=finch_2012_df)