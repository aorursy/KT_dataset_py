import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style = 'whitegrid')
import os

for dirname, _, filenames in os.walk('../input/statbunker-football-stats/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
offense = pd.concat(map(pd.read_csv, ['../input/statbunker-football-stats/Team Offense 2017-18.csv',

                                                 '../input/statbunker-football-stats/Team Offense 2018-19.csv'])).reset_index()
defense = pd.concat(map(pd.read_csv, ['../input/statbunker-football-stats/Team Defense 2017-18.csv',

                                                  '../input/statbunker-football-stats/Team Defense 2018-19.csv'])).reset_index()
tables = pd.concat(map(pd.read_csv, ['../input/statbunker-football-stats/Tables 2017-18.csv',

                                                 '../input/statbunker-football-stats/Tables 2018-19.csv'])).reset_index()
arsenal_tables = tables.loc[(tables.Team == 'Arsenal')

                            & (tables['Table Type'] == 'League Table')].set_index('Team')

arsenal_tables.drop(['index', 'KEY', 'Table Type'], axis = 1, inplace = True)

arsenal_tables
arsenal_offense = offense.loc[offense.Team == 'Arsenal'].set_index('Team')

arsenal_offense.drop(['index', 'KEY', 'GF Pld'], axis = 1, inplace = True)

arsenal_offense
arsenal_defense = defense.loc[defense.Team == 'Arsenal'].set_index('Team')

arsenal_defense.drop(['index', 'KEY', 'GA Pld'], axis = 1, inplace = True)

arsenal_defense
arsenal_stats = pd.concat([arsenal_tables, arsenal_offense, arsenal_defense], axis = 1)

arsenal_stats = arsenal_stats.loc[:,~arsenal_stats.columns.duplicated()]
pd.set_option('display.max_columns', None)

arsenal_stats
g_home = sns.catplot(x = 'GF Home', y = 'GA Home', col = 'Season',

                capsize = .2, palette = 'Reds', height = 6, aspect = .75,

                kind = 'point', data = arsenal_stats)

g_home.despine(left = True);
g_away = sns.catplot(x = 'GF Away', y = 'GA Away', col = 'Season',

                capsize = .2, palette = 'Reds', height = 6, aspect = .75,

                kind = 'point', data = arsenal_stats)

g_away.despine(left = True);
gf_melt = pd.melt(arsenal_offense,

           id_vars = ['League', 'GF Per Match', 'Season'],

           var_name = 'GF Stat')
g_gf = sns.catplot(x = 'GF Stat', y = 'value', col = 'Season',

                   data = gf_melt, kind = 'bar', palette = 'Reds')



g_gf.set_xticklabels(rotation = -85);
ga_melt = pd.melt(arsenal_defense,

           id_vars = ['League', 'GA Per Match', 'Season'],

           var_name = 'GA Stat')
g_ga = sns.catplot(x = 'GA Stat', y = 'value', col = 'Season',

                   data = ga_melt, kind = 'bar', palette = 'Reds')



g_ga.set_xticklabels(rotation = -85);