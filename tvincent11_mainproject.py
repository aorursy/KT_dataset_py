import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



PokemonFile = pd.read_csv('../input/Pokemon.csv', encoding='latin1', index_col=0)

PokemonFile   # Simple data display
PokemonFile.describe()
# Graph to dipslay the top 10 strongest Attackers

from scipy.stats import kde



Atk = PokemonFile.nlargest(10, "Attack")

Atk
# Graph to dipslay the top 10 strongest Attackers

from scipy.stats import kde



Def = PokemonFile.nlargest(10, "Defense")

Def
# Scatterplot graph of Attack and Defense

sns.lmplot(x='Attack', y='Defense', data=PokemonFile,

           fit_reg=False, # disables regression line

           hue='Stage')   # Colors for each stage
# Attack Table

plt.figure(num=None, figsize=(45,30))    # Resize the graph to make a little readable

plt.bar(x=np.arange(1,152), height=PokemonFile["Attack"], color="#D00027")

plt.xticks(np.arange(len(PokemonFile["Name"])), PokemonFile["Name"], rotation=40, fontsize=7)

plt.title("Attack Table", fontsize=15)

plt.xlabel("Names", fontsize=15)

plt.ylabel("Attack Strength", fontsize=15)

plt.show()  # Double Click table to see a little better
# Defense tabel

plt.figure(num=None, figsize=(17,36))

plt.barh(width=PokemonFile["Defense"], y=np.arange(1,152))

plt.title("Defense Table", fontsize=20)

plt.yticks(np.arange(len(PokemonFile["Name"])), PokemonFile["Name"], rotation=40, fontsize=8)

plt.xlabel("Defense Strength", fontsize=15)

plt.ylabel("Names", fontsize=15)

plt.show()
# Colors for each category

TypeColors = ['#78C850',        # Grass

                    '#F08030',  # Fire

                    '#6890F0',  # Water

                    '#A8B820',  # Bug

                    '#A8A878',  # Normal

                    '#A040A0',  # Poison

                    '#F8D030',  # Electric

                    '#E0C068',  # Ground

                    '#EE99AC',  # Fairy

                    '#C03028',  # Fighting

                    '#F85888',  # Psychic

                    '#B8A038',  # Rock

                    '#705898',  # Ghost

                    '#98D8D8',  # Ice

                    '#7038F8',  # Dragon  

             ]

              

sns.countplot(x='Type 1', data=PokemonFile, palette=TypeColors) 

plt.xticks(rotation=-45)
# Display Type 2 amount in percentage in Pie Chart



labels = "Poison", "Flying","Ground", "Fairy", "Fighting","Psychic","Steel","Ice", "Rock", "Water"

size = [19,19,6,3,1,6,2,3,2,4]

colors = ["#A040A0","#7038F8", "#E0C068", "#EE99AC", "#C03028", "#F85888", "gray","#98D8D8", "#B8A038","#6890F0"]

fig1, ax1 = plt.subplots()

ax1.pie(size, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90, colors=colors)

ax1.axis('equal')                         # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Pie Chart of Type 2", fontsize=30)

plt.show()
# Factor Plot

g = sns.factorplot(x='Type 1', 

                   y='Attack', 

                   data=PokemonFile, 

                   hue='Stage',  # Color by stage

                   col='Stage',  # Separate by stage

                   kind='swarm') # Swarmplot

 

# Rotate x-axis labels for x-axis name separation

g.set_xticklabels(rotation=-40)
# Density Plot

sns.set(style="white", color_codes=True)

sns.jointplot(x=PokemonFile["Attack"], y=PokemonFile["Defense"],

             kind="kde", color="skyblue")
# Distribution Plot of Super Attacks from pokemons

sns.distplot(PokemonFile['Total'], color='purple')
# Aggregate data points over 2D areas between super attack and super defense

sns.jointplot(PokemonFile['Sp. Atk'], PokemonFile['Sp. Def'], kind='hex', color='green')