# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

bgg_data = pd.read_csv('../input/bgg_db_2017_04.csv', header=0, encoding='latin1', index_col=0)

bgg_data.head()
bgg_data.columns
fig1, ax1 = plt.subplots(figsize=(10,6))

bgg_data.plot(x='avg_rating', y='geek_rating', ax=ax1, kind='scatter')

myX = [5.0,10.0]

myY = [5.0,10.0]

ax1.plot(myX, myY, color='red')

plt.xlim([5.55,9.5])

plt.ylim([5.55,8.5])

plt.show()
# Set the quantiles & the respective colors

quant_cat = ['0-10%','10-25%','25-50%','50-75%','75-90%','90 - 99%', '>99%']

colors = {'0-10%':'red', '10-25%':'blue', '25-50%':'green', '50-75%':'brown', 

         '75-90%':'orange','90 - 99%':'grey', '>99%':'black'}



# split by ownership!

bgOwnedQuant = bgg_data['owned'].quantile([0,0.1, 0.25, 0.5, 0.75, 0.9,0.99,1]).tolist()

bgg_data['ownership'] = pd.cut(bgg_data['owned'], bgOwnedQuant, labels=quant_cat, 

                               include_lowest=True, right=True).astype('category')

# split by votership

bgVotesQuant = bgg_data['num_votes'].quantile([0,0.1, 0.25, 0.5, 0.75, 0.9,0.99,1]).tolist()

bgg_data['votership_turnout'] = pd.cut(bgg_data['num_votes'], bgVotesQuant, labels=quant_cat, 

                               include_lowest=True, right=True).astype('category')



# Make those plots

fig, (ax1, ax2) = plt.subplots(2, figsize=(12,16))

ax1.set_title('Geek Rating vs Average Rating by Ownership Quantiles')

for label, group in bgg_data.groupby('ownership'):

    group.plot(x='avg_rating', y='geek_rating', kind="scatter", 

               marker='o', alpha=0.5, ax=ax1, color=colors[label], label=label)

ax1.plot(myX, myY, color='red')  

ax1.set_xlabel('Avg Rating')

ax1.set_ylabel('Geek Rating')



ax2.set_title('Geek Rating vs Average Rating by Votership Quantiles')

for label, group in bgg_data.groupby('votership_turnout'):

    group.plot(x='avg_rating', y='geek_rating', kind="scatter", 

               marker='o', alpha=0.5, ax=ax2, color=colors[label], label=label)

ax2.set_xlabel('Avg Rating')

ax2.plot(myX, myY, color='red')

ax2.set_ylabel('Geek Rating')

ax1.legend(loc='upper left')

ax2.legend(loc='upper left')

ax1.set_xlim([5.55,9.5])

ax2.set_xlim([5.55,9.5])

ax1.set_ylim([5.55,8.5])

ax2.set_ylim([5.55,8.5])

plt.show()
bgg_data['penetration_ratio'] = bgg_data['num_votes'] / bgg_data['owned']

bgg_most_rated_data = bgg_data.copy()

#bgg_most_rated_data = bgg_data[bgg_data['num_votes'] >= 1000]

#bgg_most_rated_data = bgg_most_rated_data[bgg_most_rated_data['votership_turnout'] == ('>99%' or '90-99%' or '75-90%' or '50-75%')]

fig3, (ax3_1, ax3_2) = plt.subplots(2, figsize=(16,12))

ax3_1.set_title('Penetration Ratio as a function of Ownership')

ax3_2.set_title('Penetration Ratio as a function of Votership')

for label, group in bgg_most_rated_data.groupby('ownership'):

    group.plot(x='owned', y='penetration_ratio', kind="scatter", 

               marker='x', alpha=0.5, ax=ax3_1, color=colors[label], label=label)

for label, group in bgg_most_rated_data.groupby('votership_turnout'):

    group.plot(x='num_votes', y='penetration_ratio', kind="scatter", 

               marker='x', alpha=0.5, ax=ax3_2, color=colors[label], label=label)

more_voters_Y = [1.0, 1.0]

more_voters_X = [5.5, 110000]

ax3_1.plot(more_voters_X, more_voters_Y, color='red')

ax3_2.plot(more_voters_X, more_voters_Y, color='red')

ax3_1.set_xlabel('Number of Owners')

ax3_1.set_ylabel('Penetration Ratio')

ax3_1.set_xscale('log')

ax3_1.set_xlim([40, 110000])

ax3_1.set_ylim([0,2])

ax3_2.set_xlabel('Number of Ratings')

ax3_2.set_ylabel('Penetration Ratio')

ax3_2.set_xscale('log')

ax3_2.set_xlim([40, 110000])

ax3_2.set_ylim([0,2])

plt.show()
bgg_data[bgg_data['penetration_ratio'] >= 1.0].loc[bgg_data['num_votes'] >= 1000, 

                                                   ['names', 'year', 'geek_rating', 'num_votes', 'penetration_ratio']].sort_values(by='penetration_ratio', ascending=False).head()
def get_categorical_data(series):#Fuction for extracting set of categorical data labels

    category_names = series.apply(lambda s:s.split(','))

    category_names = category_names.tolist()

    all_the_categories = []

    for game in category_names:

        for item in game:

            all_the_categories.append(item.replace('\n', ' ').replace('/', '-').strip())

    return set(all_the_categories)
category_set = get_categorical_data(bgg_data['category'])

mechanics_set = get_categorical_data(bgg_data['mechanic'])



{'Category':category_set, 'Mechanics':mechanics_set}
bgg_mechanics_data = bgg_data.loc[:, ['names', 'year', 'mechanic', 'geek_rating', 'avg_rating']] 

#for mech in sorted(list(mechanics_set)):

#    bgg_mechanics_data[mech] = np.zeros(len(bgg_mechanics_data.index))

for game in bgg_mechanics_data.index.tolist():

    game_mechs = bgg_mechanics_data.loc[game, 'mechanic'].split(',')

    game_mechs = [s.replace('\n', ' ').replace('/', '-').strip() for s in game_mechs]

    for mech in game_mechs:

        bgg_mechanics_data.loc[game, mech] = bgg_mechanics_data.loc[game, 'avg_rating']



#bgg_mechanics_data.describe().T.sort_values(by='count', ascending=False)

fig4, ax4 = plt.subplots(figsize=(22,10))

bgg_mechanics_data.boxplot(sorted(list(mechanics_set)), ax=ax4, showmeans=True)

ax4.set_xticklabels(sorted(list(mechanics_set)), rotation=40, ha='right')

plt.show()
bgg_top_mechanics_data = bgg_mechanics_data[bgg_mechanics_data.columns[bgg_mechanics_data.count()>200]]



top_mechanics = bgg_top_mechanics_data.columns.values.tolist()[4:]

fig5, ax5 = plt.subplots(figsize=(22,10))

bgg_top_mechanics_data.boxplot(top_mechanics, ax=ax5, showmeans=True)

ax5.set_xticklabels(top_mechanics, rotation=40, ha='right')

plt.show()
bgg_top_mechanics_data.describe().T.sort_values(by='mean', ascending=False)
bgg_cat_data = bgg_data.loc[:, ['names', 'year','owned', 'category','geek_rating', 'avg_rating' ]] 

for game in bgg_cat_data.index.tolist():

    game_cats = bgg_cat_data.loc[game, 'category'].split(',')

    game_cats = [s.replace('\n', ' ').replace('/', '-').strip() for s in game_cats]

    for cat in game_cats:

        bgg_cat_data.loc[game, cat] = bgg_cat_data.loc[game, 'avg_rating']



bgg_top_cat_data = bgg_cat_data[bgg_cat_data.columns[bgg_cat_data.count()>200]]



top_cat = bgg_top_cat_data.columns.values.tolist()[5:]

fig6, ax6 = plt.subplots(figsize=(22,10))

bgg_top_cat_data.boxplot(top_cat, ax=ax6, showmeans=True)

ax6.set_xticklabels(top_cat, rotation=40, ha='right')

plt.show()
bgg_top_cat_data.describe().T
import math 

from scipy.stats import ks_2samp

miniatures_comp = bgg_cat_data.loc[:, ['names', 'year','owned', 'category','geek_rating', 'avg_rating', 'Miniatures' ]] 

for game in miniatures_comp.index.tolist():

    #print(game['Miniatures'])

    if math.isnan(miniatures_comp.loc[game, 'Miniatures']) :

        miniatures_comp.loc[game, 'No Miniatures'] = miniatures_comp.loc[game,'avg_rating']

# miniatures_comp[['Miniatures', 'No Miniatures']].describe()



fig7, ax7 = plt.subplots()

mini_heights, mini_bins = np.histogram(miniatures_comp['Miniatures'].dropna(axis=0), bins=50)

nomini_heights, nomini_bins = np.histogram(miniatures_comp['No Miniatures'].dropna(axis=0), bins=mini_bins)

mini_heights = mini_heights / miniatures_comp['Miniatures'].count()

nomini_heights = nomini_heights / miniatures_comp['No Miniatures'].count()

width = (mini_bins[1] - mini_bins[0])/3

ax7.bar(mini_bins[:-1], mini_heights, width=width, facecolor='cornflowerblue', label='Minis')

ax7.bar(nomini_bins[:-1]+width, nomini_heights, width=width, facecolor='seagreen', label='No Minis')

ax7.set_xlabel('Average Rating')

ax7.set_ylabel('Rating Denisty')

ax7.legend(loc='upper right')

plt.show()



# do Kolmogorov - Smirnov analysis of null hypothesis



print(ks_2samp(miniatures_comp['Miniatures'].dropna(axis=0),

               miniatures_comp['No Miniatures'].dropna(axis=0)))
