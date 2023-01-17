# Imports 

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.gridspec as gridspec
data = pd.read_csv('../input/NBA_player_of_the_week.csv', index_col = 'Date', parse_dates = True)

data.head()
unique_heights = data.Height.unique()

unique_heights.sort()

unique_heights
unique_weights = data.Weight.unique()

unique_weights.sort()

unique_weights
print(data.shape)

data.Conference.value_counts()
data.Conference.isnull().astype(int).plot()

plt.title('Missing value for conference by award date')

plt.show()
max(data[data.Conference.isnull()].index)
def clean_heights(height):

    '''

    Converts the height column to a common numeric format and unit (cm)

    '''

    total_inches = 0

    

    # There were initially a mixture of values so this just a try catch 

    if height.find('cm') != -1:

        return int(height.replace('cm', ''))

    else:

        feet = int(height.split('-')[0])

        inches = int(height.split('-')[1])

        total_inches += feet * 12

        total_inches += inches



    return total_inches * 2.54
data.Height = data.Height.apply(clean_heights)
fig, axes = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (12, 4))

sns.set_style("whitegrid")



plt.sca(axes[0])

sns.distplot(data.Height, color = '#27ae60')

plt.title('Distribution of player heights (cm)')



plt.sca(axes[1])

sns.distplot(data.Weight, color = '#e67e22')

plt.title('Distribution of player weights (lb)')

plt.show()
g = sns.jointplot(x=data.Height, y=data.Weight, kind = 'reg', marginal_kws={'color': '#27ae60'})

plt.setp(g.ax_marg_y.patches, color='#e67e22')

plt.setp(g.ax_marg_y.lines, color = '#e67e22')

plt.show()
figure = plt.figure(figsize = (10,10))

sns.scatterplot(x = data.Height, y = data.Weight, hue = data.Position)

plt.show()
g = sns.FacetGrid(data, col = 'Position', col_wrap=4, hue='Position')

g.map(sns.scatterplot, 'Height', 'Weight')

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Heights and Weights of players by position')

plt.show()
data.Player.value_counts().describe()
# Set some plotting parameters

plt.rcParams['figure.figsize']=(10, 6)

plt.style.use('fivethirtyeight')



sns.distplot(data.Player.value_counts(), rug = True, color = 'y')

plt.xlabel('Number of awards for a single player')

plt.title('Distribution of "Player of the week" awards')

plt.axvline(data.Player.value_counts().max(), color = 'purple')

plt.annotate(data.Player.value_counts().idxmax(), xy = (61, 0.4), bbox=dict(boxstyle="round", fc="none", ec="purple"), xytext = (-125, 0), textcoords='offset points')

plt.show()
data.Player.value_counts().head(10).plot.bar(fontsize = 10)

plt.title('Players with the most "Player of the week" awards')

plt.xticks(rotation = 45)

plt.show()
plt.boxplot(data.Age, vert = False)

plt.title("Distribution of Age of 'Player of the week' winners")

plt.xlabel("Age of award winner")

plt.yticks([])

plt.show()
sns.distplot(data.Age)

plt.title("Distribution of Age of 'Player of the week' winners")

plt.xlabel("Age of award winner")

plt.axvline(data.Age.mean(), color = 'red')

plt.show()
lebron = data[data['Player'] == 'LeBron James'].sort_index()

lebron.groupby('Season').size().plot.bar()

plt.xticks(rotation = 60)

plt.title('LeBron James awards by NBA Season')

plt.show()
data['Seasons in league'].value_counts().sort_index().plot.bar()

plt.xticks(rotation = 'horizontal')

plt.xlabel('Seasons in the league of winner')

plt.title('League experience vs. awards given')

plt.show()