# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pokemon_df = pd.read_csv('../input/pokemon.csv')
combats_df = pd.read_csv('../input/combats.csv')
pokemon_df.info()
pokemon_df[['Type 1', 'Type 2']].isnull().sum()
possible_types = set(pokemon_df['Type 1'])
matches_by_type = pd.DataFrame(0, index=possible_types, columns=possible_types, dtype=int)
wins_by_type = pd.DataFrame(0, index=possible_types, columns=possible_types, dtype=int)

# join the types to the combats table to make it a little easier. 
combats_df = pd.merge(combats_df, pokemon_df, left_on='First_pokemon', right_on='#', how='left')
combats_df = combats_df.rename(mapper={"#":"First_pokemon #", "Name":"First_pokemon Name", "Type 2": "First_pokemon Type 2", "Type 1":"First_pokemon Type 1","HP": "First_pokemon HP", "Attack": "First_pokemon Attack", 'Defense': "First_pokemon Defense", 'Sp. Atk': "First_pokemon Sp.Atk", 'Sp. Def': "First_pokemon Sp.Def", 'Speed': "First_pokemon Speed", 'Generation': "First_pokemon Generation", 'Legendary': "First_pokemon Legendary"}, axis='columns')

combats_df = pd.merge(combats_df, pokemon_df, left_on='Second_pokemon', right_on='#', how='left')
combats_df = combats_df.rename(mapper={"#":"Second_pokemon #", "Name":"Second_pokemon Name", "Type 2": "Second_pokemon Type 2", "Type 1":"Second_pokemon Type 1","HP": "Second_pokemon HP", "Attack": "Second_pokemon Attack", 'Defense': "Second_pokemon Defense", 'Sp. Atk': "Second_pokemon Sp.Atk", 'Sp. Def': "Second_pokemon Sp.Def", 'Speed': "Second_pokemon Speed", 'Generation': "Second_pokemon Generation", 'Legendary': "Second_pokemon Legendary"}, axis='columns')

combats_df
for index, row in combats_df.iterrows():
    p1_type = row['First_pokemon Type 1']
    p2_type = row['Second_pokemon Type 1']
    matches_by_type[p1_type][p2_type] = matches_by_type[p1_type][p2_type] + 1
    if (row['First_pokemon'] == row['Winner']):
        wins_by_type[p1_type][p2_type] = wins_by_type[p1_type][p2_type] + 1
wins_by_type_probabilities = wins_by_type / matches_by_type 
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(wins_by_type_probabilities, vmax=1.0,square=True);
d = combats_df.loc[(combats_df['First_pokemon Type 1'] == "Electric") & (combats_df['Second_pokemon Type 1'] == "Fairy")]
d['first_won'] = d['First_pokemon'] == d['Winner']
print( d['first_won'].sum() / d['first_won'].count())
print (wins_by_type_probabilities['Electric']['Fairy'])
wins_by_type_probabilities.mean().sort_values(ascending=False)
combats_df['First_pokemon won'] = (combats_df['First_pokemon'] == combats_df['Winner']).map({True: 1, False: 0})
combats_df
def win_percentage_by_stat(first_stat, second_stat, stat_name):
    combats_df['diff'] = combats_df[first_stat] - combats_df[second_stat]
    stat_bins = pd.qcut(combats_df['diff'], 10)
    bin_col = stat_name + " Bins"
    d = pd.DataFrame({bin_col: stat_bins, "First Won": combats_df['First_pokemon won']})
    bins = sorted(set(d[bin_col]))

    percentages = []
    for b in bins:
        bin_rows = d.loc[d[bin_col] == b]
        win_percentage = (bin_rows['First Won'] == 1).sum() / bin_rows['First Won'].count()
        percentages.append( win_percentage )

    results = pd.DataFrame({bin_col: bins, "Win Percentage": percentages})
    return results
results = win_percentage_by_stat("First_pokemon HP", "Second_pokemon HP", "HP")
print("Spread:" + str(max(results['Win Percentage']) - min(results['Win Percentage'])))
results
results = win_percentage_by_stat("First_pokemon Attack", "Second_pokemon Attack", "Attack")
print("Spread:" + str(max(results['Win Percentage']) - min(results['Win Percentage'])))
results
results = win_percentage_by_stat("First_pokemon Defense", "Second_pokemon Defense", "Defense")
print("Spread:" + str(max(results['Win Percentage']) - min(results['Win Percentage'])))
results
results = win_percentage_by_stat("First_pokemon Sp.Atk", "Second_pokemon Sp.Atk", "Sp. Atk")
print("Spread:" + str(max(results['Win Percentage']) - min(results['Win Percentage'])))
results
results = win_percentage_by_stat("First_pokemon Attack", "Second_pokemon Defense", "Atk - Def")
print("Spread:" + str(max(results['Win Percentage']) - min(results['Win Percentage'])))
results
