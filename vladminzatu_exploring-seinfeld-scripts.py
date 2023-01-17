import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_colwidth', -1) # I want to see the full column contents when displaying the dataframes
scripts = pd.read_csv('../input/scripts.csv')

episode_info = pd.read_csv('../input/episode_info.csv')
scripts.head()
scripts[scripts.isnull().any(axis=1)]
scripts = scripts.dropna(axis=0)

scripts.info()
scripts['Season'].unique()
sns.distplot(scripts['Season'], kde=False)
episodes_per_season = scripts.groupby('Season')['SEID'].aggregate(['count', 'unique'])

episodes_per_season['nr_episodes'] = episodes_per_season['unique'].apply(lambda x: len(x))

episodes_per_season[['count', 'nr_episodes']]
episodes_per_season['count']/episodes_per_season['nr_episodes']
scripts[scripts['Season'] == 1.0]['SEID'].unique()
def lines_in_episode(episode):

    return scripts[scripts['SEID'] == episode]['Dialogue'].count()



print('# lines S01E01: %d . # lines in S01E02: %d' %(lines_in_episode('S01E01'),lines_in_episode('S01E02')))
scripts[205:215]
scripts[:211]['SEID'] = 'Pilot'
scripts['Character'].value_counts()
def plot_lines(season = None, episode = None, top_n = 10, ax = None):

    filtered_scripts = scripts

    if season:

        filtered_scripts = filtered_scripts[filtered_scripts['Season'] == season]

    if episode:

        filtered_scripts = filtered_scripts[filtered_scripts['SEID'] == episode]

    filtered_scripts['Character'].value_counts().head(top_n).plot(kind = 'bar', ax = ax)
plot_lines()
scripts[scripts['Character'] == '[Setting'].head(10)
fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(10, 20), dpi=100)

for i in range(9):

    season = i + 1

    row = i//3

    col = i%3

    plot_lines(season = season, ax = axes[row][col])

    axes[row][col].set_title(f'Season {season}', fontsize=12)

plt.show()
scripts['word_count'] = scripts['Dialogue'].apply(lambda x: len(str(x).split()))

scripts['mean_word_length'] = scripts['Dialogue'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
def get_characters_having_lines(min_lines):

    character_lines = scripts['Character'].value_counts()

    characters_with_multiple_lines = character_lines.index[character_lines > min_lines].tolist()

    return characters_with_multiple_lines
characters = get_characters_having_lines(100)

characters
# Following the example in https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function

def get_percentiles(column_name):

    def percentile(n):

        def percentile_(x):

            return np.percentile(x, n)

        percentile_.__name__ = 'percentile_%s' % n

        return percentile_

    return scripts[scripts['Character'].isin(characters)][['Character', column_name]].groupby('Character').agg(

        [percentile(50), percentile(75), percentile(95)])
get_percentiles('word_count')
get_percentiles('mean_word_length')