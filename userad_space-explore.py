#Ok, standart header



import numpy as np

import pandas as pd 

from matplotlib import pyplot as plt

import matplotlib as mpl



import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline

mpl.rc("figure", figsize=(12, 8))

plt.figure(figsize=(12, 8))





data=pd.read_csv('../input/oec.csv',sep=',')
data.query('DiscoveryYear > 1990')["DiscoveryYear"].plot.hist(bins=30, title="Discoveries since 1990")
data.query("HostStarTempK < 15000")["HostStarTempK"].plot.hist(bins=15, title="Temperature distribution")
#Is any correlations between host star temperature and distance? In "near" space

planets_with_data_of_distance_and_temp = data[["DistFromSunParsec", "HostStarTempK", "DiscoveryMethod"]].query("DistFromSunParsec < 500").query('HostStarTempK < 8000')

planets_with_data_of_distance_and_temp.dropna()

planets_with_data_of_distance_and_temp.plot.scatter(x="DistFromSunParsec", y="HostStarTempK", figsize=(12,8))