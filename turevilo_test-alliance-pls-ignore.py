# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv("../input/vgsales.csv")

df = df[df["Year"] <=2017]

#df.info()

df.head(10)
print("Number of games:", len(df))

print("Number of genres:", len(df["Genre"].unique()))

print("Number of publishers:", len(df["Publisher"].unique()))

print("Number of platforms:", len(df["Platform"].unique()))
df.Year.plot.hist(bins=[1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020],

                  grid=True, rwidth=0.95, figsize=(10, 6), fontsize=16)



plt.grid(alpha=0.75, dashes=(7,7))

plt.title("Games released (1980-...)", fontsize=20)

plt.ylabel("Titles", fontsize=12)

plt.gca().xaxis.grid(False) #for removal of horizontal gridlines

plt.show()
df.plot.scatter("Year", "Global_Sales", alpha=0.5, grid=True, figsize=(10,6), fontsize=16)



plt.grid(alpha=0.75, dashes=(7,7))

plt.title("Games sold (1980-...)", fontsize=20)

plt.xlabel("")

plt.ylabel("Copies (in millions)", fontsize=12)

plt.gca().xaxis.grid(False) #for removal of horizontal gridlines

plt.show()
PuGlo = df.groupby("Publisher")["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"].sum().reset_index().sort_values("Global_Sales",ascending=False)

PuGlo.index = pd.RangeIndex(1, len(PuGlo)+1)

PuGlo.head(20)
sold = {}

sold["Region"]=["EU_Sales", "NA_Sales", "JP_Sales", "Other_Sales"]

sold["Copies"]=[]

for area in sold["Region"]:

    sold["Copies"] += [round(df[area].sum(),2)]

soldDF = pd.DataFrame(data=sold)

soldDF = soldDF[["Region","Copies"]]
soldDF["Copies"].plot.pie(labels=["EU","NA","JP","Other"], figsize=(8,8), fontsize=28, autopct="%.1f%%")

plt.ylabel("")

plt.title("Sales", fontsize=20)

plt.show()
PlVal = df["Platform"].value_counts()

PlVal.plot.barh(figsize=(11,10), grid=True, width=0.9, fontsize=14)



plt.grid(alpha=0.75, dashes=(12,12))

plt.gca().yaxis.grid(False) #for removal of horizontal gridlines

plt.title("Amount of titles on every platform", fontsize=20)

plt.ylabel("Platform", fontsize=14)

plt.xlabel("Number of games released", fontsize=14)

plt.show()
av = PlVal.mean()

PlVal2 = {"Others":0}

for key in PlVal.keys():

    value = PlVal[key]

    if value >= av:

        PlVal2[key] = value

    else:

        PlVal2["Others"] += value



PlVal2a = {"Platform":[], "Titles":[]}

for key in PlVal2.keys():

    PlVal2a["Platform"] += [key]

    PlVal2a["Titles"] += [PlVal2[key]]
PlVal2b = pd.DataFrame(data=PlVal2a)

PlVal2b = PlVal2b[["Platform","Titles"]]

PlVal2b["Titles"].plot.pie(labels=PlVal2b["Platform"], figsize=(14,14), fontsize=20, autopct="%.1f%%")

plt.title("Platforms with most titles released", fontsize=20)

plt.ylabel("")

plt.show()