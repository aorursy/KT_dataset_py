import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style="whitegrid")
df_coal = pd.read_csv("../input/production.csv")

df_coal.head(2)
df_india = df_coal[df_coal["States"] == "India"].nlargest(10,'Quantity 2014-15(P)')

df_india.head(2)
f, ax = plt.subplots(figsize=(9, 3))

df_india[["Quantity 2010-11","Quantity 2011-12", "Quantity 2012-13", "Quantity 2013-14", "Quantity 2014-15(P)"]].plot.bar(ax=ax)

f.suptitle('Top 10 Minerals (India)', fontsize=14)

plt.xticks(rotation=60)

plt.xlabel('Minerals', fontsize=12)

wrap = ax.set_xticklabels(list(df_india["Mineral"]))
df = df_coal.drop(df_coal[df_coal["States"] == "India"].index).reset_index()

df = df.drop(['index'], axis=1)

df = df.fillna(value=0) # make all missing values = 0

df.head(2)
top5States = list(df["States"].value_counts().nlargest(5).index)

top5States
df_top5States = df[df['States'].isin(top5States)]

df_top5States_pivot = pd.pivot_table(df_top5States,index=["States", "Mineral"],

               values=["Quantity 2010-11","Quantity 2011-12", "Quantity 2012-13", "Quantity 2013-14", "Quantity 2014-15(P)"])

df_top5States_pivot.head()
f, ax = plt.subplots(figsize=(9, 3))

df_top5States_pivot.nlargest(10,'Quantity 2014-15(P)').plot.bar(ax=ax)

f.suptitle('Top 10 Minerals (Top States)', fontsize=14)

plt.xlabel('Minerals', fontsize=12)

wrap = plt.xticks(rotation=65)
for state in top5States:

    f, ax = plt.subplots(figsize=(9, 3))

    queryText = "States == ['" + state + "']"

    df_top5States_pivot.query(queryText).nlargest(5,'Quantity 2014-15(P)').plot.bar(ax=ax)

    f.suptitle('Top 5 Minerals (State: ' + state + ')', fontsize=14)

    plt.xlabel('Minerals', fontsize=12)

    wrap = plt.xticks(rotation=40)