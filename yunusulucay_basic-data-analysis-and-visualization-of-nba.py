import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



import os

print(os.listdir("../input"))
data = pd.read_csv("../input/NBA_player_of_the_week.csv")
# First 5 rows of data.

data.head()
data.info()
# Dropping Nan values

data.dropna(inplace=True)
# I've dropped some columns bec. of that index is not starting like 0 1 2 maybe 356 389 ...

# So i will sort and make the index to start 0 with arange.

data.index = np.arange(0,len(data))

data["Active season"].value_counts()
data.drop(['Active season','Real_value'],axis=1,inplace=True)

data.head()
# Correlation's heatmap.

f,ax = plt.subplots(figsize=(10,8))

sns.heatmap(data.corr(),annot=True, linewidths=.5,fmt=".1f",ax=ax)

plt.show()
data.Age.plot(kind = 'line',color = 'black',label = 'AGE',linewidth=1,alpha=0.6,grid=True,linestyle = '-')

data["Seasons in league"].plot(color = "blue",label = "SEASONS IN LEAGUE",linewidth=1,alpha=0.6,grid=True,linestyle="-.")

plt.legend(loc="upper right")

plt.xlabel("Age & Seasons in League")

plt.ylabel("Value")

plt.title("Age - Seasons in League Graphic")

plt.show()
data.plot(kind = "scatter",x = "Age", y = "Seasons in league", alpha = 0.5, color="Black")

plt.xlabel("Age")

plt.ylabel("Seasons in league ")

plt.title("Analyzing on Scatter")

plt.show()

plt.savefig("Graphic.png")
data.Age.plot(kind = 'hist',bins = 50,figsize = (10,8))

plt.show()
threshold = sum(data.Age)/len(data.Age)

data["Age Comparison"] = ["High" if i > threshold else "Low" for i in data.Age]

data.loc[:10,["Age Comparison","Age"]]
# Players that have high Seasons in League

bestplayers=data[data["Seasons in league"] == max(data["Seasons in league"])]["Player"]

bestplayers.index = np.arange(0,len(bestplayers))

print("1.player is :"+bestplayers[0]+"\n2.player is :"+bestplayers[1])