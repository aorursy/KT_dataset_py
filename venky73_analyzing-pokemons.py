# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Pokemon.csv")
df.head()
df.describe()
df["Attack"] = df["Attack"] + df["Sp. Atk"]

df["Defense"] = df["Defense"] + df["Sp. Def"]
df.head()
df.drop(columns=["Sp. Atk","Sp. Def",], inplace=True)
df.head()
df.isna().sum()
df.drop(columns=["Type 2","#"], inplace=True)
df.head()
df["Type 1"].value_counts()
df.Generation.value_counts()
sns.barplot(x="Type 1",y = "Total", data = df).set_title("Type 1 vs Avg-score")

plt.xticks(rotation = 90)
sns.barplot(x="Generation",y = "Total", data = df).set_title("Generation vs Avg-score")
sns.barplot(x="Type 1",y = "Speed", data = df).set_title("Type 1 vs Avg-Speed-score")

plt.xticks(rotation = 90)
sns.barplot(x="Type 1",y = "HP", data = df).set_title("Type 1 vs Avg-HP-score")

plt.xticks(rotation = 90)
sns.barplot(x="Generation",y = "Speed", data = df).set_title("Generaion vs Avg-Speed-score")
sns.barplot(x="Generation",y = "HP", data = df).set_title("Generation vs Avg-HP-score")
plt.figure(figsize=(15,8))

sns.barplot(x = "Type 1", y = "Total",hue = "Generation", data = df )

plt.xticks(rotation = 90)

df.groupby(["Type 1","Generation"])["Total"].mean()
print("\n best pokemons based on Total Scores of their respective Generations\n")

ref = dict(df.groupby(["Generation"])["Total"].max())

for i in range(1,7):

    print("\nGeneration : "+str(i)+": "+str(df[(df.Generation == i) & (df.Total == ref[i])].Name))
print("\n best pokemons based on Total Scores of their respective Types\n")

ref = dict(df.groupby(["Type 1"])["Total"].max())

for key,value in ref.items():

    print("best pokemon based on Type 1 : "+key+" : ",df[(df["Type 1"] == key) & (df.Total == value)].Name)
print("Top 5 Worst Pokemons based on Total scores")

df.sort_values(by = "Total")[:5]
print("Top 5 Best Pokemons based on Total scores")

df.sort_values(by = "Total",ascending = False)[:5]