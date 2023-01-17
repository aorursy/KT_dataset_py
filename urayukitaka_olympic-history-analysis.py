# Basic libraries

import numpy as np

import pandas as pd



# Directry check

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Visualization

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()
df = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv", header=0)
df.head()
# data size

df.shape
# data info

df.info()
# Null check

df.isnull().sum()
# Unique values

def unique_values(data):

    for i in range(data.shape[1]):

        print("data size:{}".format(data.shape[0]))

        print(data.columns[i])

        print(data.iloc[:,i].value_counts())

        print("-"*40)



unique_values(df)
# Check for duplicate data

print("df_shape:", df.shape)

print("df_unique_shape", df.drop_duplicates().shape)
df.drop_duplicates(inplace=True)
# group by sex

data = pd.DataFrame(data=df.groupby("Season").ID.count()).reset_index()



# Visualization

plt.figure(figsize=(6,6), facecolor='lavender')

plt.pie(data["ID"], labels=data["Season"], shadow=True, autopct='%1.1f%%')

plt.title("Ratio of Summer & Winter")
# data

pivot = pd.pivot_table(df, index="Year", columns="Season", values="ID", aggfunc="count").reset_index()



# Visualization

plt.figure(figsize=(10,6), facecolor='lavender')

plt.plot(pivot["Year"], pivot["Summer"].fillna(method='ffill'), color="red")

plt.plot(pivot["Year"], pivot["Winter"].fillna(method='ffill'), color="blue")

plt.xlim([1890,2020])

plt.xlabel("Year")

plt.ylabel("Counts of players")

plt.legend(labels=["Summer", "Winter"], facecolor="white")

plt.title("Graph of the number of participants by year in summer and winter")
# data

group = pd.DataFrame(data=df.groupby("City").ID.count()).reset_index().sort_values(by="ID", ascending=False)



index = group["City"]

value = group["ID"]



# Visualization

plt.figure(figsize=(20,6),facecolor='lavender')

plt.bar(index, value, alpha=0.5)

plt.xticks(rotation=90)

plt.ylabel("Counts of players")

plt.title("Participants by venue")
# data

group = pd.DataFrame(data=df.groupby("NOC").ID.count()).reset_index().sort_values(by="ID", ascending=False).head(100)



index = group["NOC"]

value = group["ID"]



# Visualization

plt.figure(figsize=(20,6),facecolor='lavender')

plt.bar(index, value, alpha=0.5)

plt.xticks(rotation=90)

plt.ylabel("Counts of players")

plt.title("Participation by country(Top100)")
# group by sex

data = pd.DataFrame(data=df.groupby("Sex").ID.count()).reset_index()



# Visualization

plt.figure(figsize=(6,6), facecolor='lavender')

plt.pie(data["ID"], labels=data["Sex"], shadow=True, autopct='%1.1f%%')

plt.title("Ratio of Sex")
plt.figure(figsize=(10,6), facecolor='lavender')

sns.distplot(df["Age"], kde=False)

plt.title("Age distribution")

plt.xlabel("Age")

plt.ylabel("count")

plt.yscale("log")
df.query("Age>=80")
plt.figure(figsize=(15,9), facecolor='lavender')



sns.jointplot(df["Weight"], df["Height"], kind="hex")

plt.title("Weight vs Height plot")
df[["Weight", "Height"]].describe()
# data

group = pd.DataFrame(data=df.query("Medal=='Gold'").groupby("NOC").ID.count()).reset_index().sort_values(by="ID", ascending=False).head(100)

index = group["NOC"]

value = group["ID"]



# Visualization

plt.figure(figsize=(20,6),facecolor='lavender')

plt.bar(index, value, alpha=0.5)

plt.xticks(rotation=90)

plt.ylabel("Counts of players")

plt.title("Gold medal ranking(Top100)")
# data

medal = pd.get_dummies(df["Medal"])

df["Gold"] = medal["Gold"]

df["Silver"] = medal["Silver"]

df["Bronze"] = medal["Bronze"]



Total_number = pd.DataFrame(data=df.groupby("NOC").ID.count()).reset_index()

Gold_count = pd.DataFrame(data=df.groupby("NOC").Gold.sum()).reset_index()



agg = pd.merge(Total_number, Gold_count, on="NOC", how="left")

agg["ratio"] = agg["Gold"]/agg["ID"]*100



index = agg.sort_values(by="ratio", ascending=False)["NOC"].head(100)

value = agg.sort_values(by="ratio", ascending=False)["ratio"].head(100)



# Visualization

plt.figure(figsize=(20,6),facecolor='lavender')

plt.bar(index, value, alpha=0.5)

plt.xticks(rotation=90)

plt.ylabel("Ratio of get gold")

plt.title("Gold medal acquisition rate(Top100)")
# data

group = pd.DataFrame(data=df.query("Sport=='Athletics' & Medal=='Gold'").groupby("NOC").ID.count()).reset_index().sort_values(by="ID", ascending=False)

index = group["NOC"]

value = group["ID"]



# Visualization

plt.figure(figsize=(20,6),facecolor='lavender')

plt.bar(index, value, alpha=0.5)

plt.xticks(rotation=90)

plt.ylabel("Counts of players")

plt.title("Athletics, Gold medal ranking")
# data

group = pd.DataFrame(data=df.query("Sport=='Swimming' & Medal=='Gold'").groupby("NOC").ID.count()).reset_index().sort_values(by="ID", ascending=False)

index = group["NOC"]

value = group["ID"]



# Visualization

plt.figure(figsize=(20,6),facecolor='lavender')

plt.bar(index, value, alpha=0.5)

plt.xticks(rotation=90)

plt.ylabel("Counts of players")

plt.title("Swimming, Gold medal ranking")
# data

group = pd.DataFrame(data=df.query("Sport=='Judo' & Medal=='Gold'").groupby("NOC").ID.count()).reset_index().sort_values(by="ID", ascending=False)

index = group["NOC"]

value = group["ID"]



# Visualization

plt.figure(figsize=(20,6),facecolor='lavender')

plt.bar(index, value, alpha=0.5)

plt.xticks(rotation=90)

plt.ylabel("Counts of players")

plt.title("Judo, Gold medal ranking")