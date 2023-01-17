import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import cm

import numpy as np
poke = pd.read_csv("../input/pokemon/pokemon.csv", header=0, index_col=0, encoding='utf-8')
poke.head(10)
poke = poke.set_index('name')
legendary = poke.groupby("is_legendary").aggregate({"is_legendary": "count"}).rename(columns={'is_legendary': 'count', 'index': 'is_legendary'})

explode = [0.15, 0]

print(legendary.reset_index())



plt.figure(figsize=(9, 10))

colors = ['skyblue', 'deepskyblue']

plt.title(label=" Legendary Pokemon in %", loc="center", fontsize=25)

patches, l_text, p_text = plt.pie(legendary["count"],

        labels=['not legendary', 'legendary'], autopct='%1.1f%%', colors=colors, explode=explode, startangle=60)



for t in p_text:  # set text size

    t.set_size(20)

for t in l_text:

    t.set_size(20)



plt.show()
base = poke[["base_total"]].sort_values('base_total', ascending=False).reset_index()

plt.figure(figsize=(10, 8))

plt.hist(base["base_total"], color='powderblue', bins=50)

plt.title(label=" Total_Points_Density", loc="center", fontsize=22)



plt.show()
plt.figure(figsize=(10, 8))

total = poke[["base_total"]].sort_values('base_total', ascending=False).reset_index()

colors = cm.Paired(np.linspace(0, 1, 10))

plt.bar(total["name"].head(10), total["base_total"].head(10), color=colors, width=0.6)



for a, b in zip(total["name"].head(10), total["base_total"].head(10)):  # display number

    plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)

plt.ylim(0, 1200)

plt.title(label=" Top10_Total_Points", loc="center", fontsize=22)



plt.show()
plt.figure(figsize=(14, 8))

ad = poke[["attack", "defense"]].reset_index()



plt.subplot2grid((1,2), (0,0))

plt.hist(ad["attack"], color='powderblue', bins=30)

plt.title(label=" Attack_value_Density", loc="center", fontsize=20)



plt.subplot2grid((1,2), (0,1))

plt.hist(ad["defense"], color='powderblue', bins=30)

plt.title(label=" Defense_value_Density", loc="center", fontsize=20)



plt.show()
ad = poke[["attack", "defense"]].reset_index()



plt.figure(figsize=(15, 8))

plt.subplot2grid((1, 2), (0, 0))

plt.scatter(ad["attack"], ad["defense"], color='powderblue')

plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title(label=" Attack & Defense", loc="center", fontsize=20)



plt.subplot2grid((1,2), (0,1))

plt.hexbin(ad["attack"], ad["defense"], cmap='PuBu', gridsize=20)

plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title(label=" Attack & Defense", loc="center", fontsize=20)



plt.show()
from sklearn.linear_model import LinearRegression



model = LinearRegression()

ad = poke[["attack", "defense"]].reset_index()



X = ad[['attack']]

Y = ad[["defense"]]



model.fit(X, Y)

Y_pred = model.predict(X)



plt.figure(figsize=(10, 8))

plt.scatter(X, Y, color='powderblue')

plt.plot(X, Y_pred, color = 'red')



plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title(label=" Attack & Defense", loc="center", fontsize=20)



plt.show()



print("R^2: ", model.score(X, Y))
att = ad.loc[ad['attack'].idxmax()]

deff = ad.loc[ad['defense'].idxmax()]



print(att)

print(deff)
wh = poke[["weight_kg", "height_m", "hp", "speed"]].dropna().reset_index()



plt.figure(figsize=(16, 8))

plt.subplot2grid((1, 2), (0, 0))

plt.hist(wh["weight_kg"], color='powderblue', bins=30)

plt.ylabel("Frequency")

plt.title(label="Weight", loc="center", fontsize=20)



plt.subplot2grid((1,2), (0,1))

plt.hist(wh["height_m"], color='powderblue', bins=30)

plt.ylabel("Frequency")

plt.title(label="Height", loc="center", fontsize=20)



plt.show()
weight_max = wh.loc[wh['weight_kg'].idxmax()]

print(weight_max)



height_max =wh.loc[wh['height_m'].idxmax()]

print(height_max)



weight_min = wh.loc[wh['weight_kg'].idxmin()]

print(weight_min)



height_min =wh.loc[wh['height_m'].idxmin()]

print(height_min)
plt.figure(figsize=(16, 8))

plt.subplot2grid((1,2), (0,0))

plt.hist(wh["hp"], color='powderblue', bins=30)

plt.ylabel("Frequency")

plt.title(label="Hp", loc="center", fontsize=20)



plt.subplot2grid((1,2), (0,1))

plt.hist(wh["speed"], color='powderblue', bins=30)

plt.ylabel("Frequency")

plt.title(label="Speed", loc="center", fontsize=20)



plt.show()
hp_max = wh.loc[wh['hp'].idxmax()]

print(hp_max)



speed_max =wh.loc[wh['speed'].idxmax()]

print(speed_max)



hp_min = wh.loc[wh['hp'].idxmin()]

print(hp_min)



speed_min =wh.loc[wh['speed'].idxmin()]

print(speed_min)
type = poke[["type1"]].reset_index().groupby("type1").aggregate({"name": "count"})

type = type.sort_values('name', ascending=False).reset_index().rename(columns={'name': 'count'})

print(type)



plt.figure(figsize=(12, 8))

colors = cm.Paired(np.linspace(0, 1, 17))

plt.bar(type["type1"], type["count"], color=colors, width=0.6)

plt.ylabel("count")

for a, b in zip(type["type1"], type["count"]):  # display number

    plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)

plt.title(label=" Primary Type", loc="center", fontsize=22)



plt.tight_layout()

plt.show()
type2 = poke[["type2"]].reset_index().dropna().groupby("type2").aggregate({"name": "count"})

type2 = type2.sort_values('name', ascending=False).reset_index().rename(columns={'name': 'count'})



plt.figure(figsize=(12, 8))

colors = cm.Paired(np.linspace(0, 1, 19))

plt.bar(type2["type2"], type2["count"], color=colors, width=0.6)

plt.ylabel("count")

for a, b in zip(type2["type2"], type2["count"]):  # display number

    plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)

plt.title(label=" Secondary Type", loc="center", fontsize=22)



plt.tight_layout()

plt.show()
gen = poke.reset_index().groupby("generation").aggregate({"name": "count"})

gen = gen.reset_index().rename(columns={'name': 'count'})



plt.figure(figsize=(10, 8))

colors = cm.Paired(np.linspace(0, 0.8, 7))

plt.bar(gen["generation"], gen["count"], color=colors, width=0.6)

plt.ylabel("count")

plt.xlabel("generation")

for a, b in zip(gen["generation"], gen["count"]):  # display number

    plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)

plt.title(label=" Number of Pokemons per Generation", loc="center", fontsize=22)



plt.show()
genl = poke.reset_index().groupby(["generation", "is_legendary"]).aggregate({"name": "count"})

genl = genl.reset_index().rename(columns={'name': 'count'})



plt.figure(figsize=(10, 8))

colors = ['lightblue', 'khaki']

plt.bar(genl["generation"], genl["count"], color=colors, width=0.6)

plt.ylabel("count")

plt.xlabel("generation")

for a, b in zip(genl["generation"], genl["count"]):  # display number

    plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)

plt.title(label=" Legendary and Non-legendary per Generation", loc="center", fontsize=20)



plt.show()