# Import libs

from glob import glob

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Set the width to display

pd.set_option('display.width', 120)

# Increase the number of rows to display

pd.set_option('display.max_rows', 60) 



# Get the data

happiness = pd.read_csv('../input/world-happiness/2016.csv')



# Column map to rename

cols_dict = {'Country':'country',

             'Country or region':'country',

             'Region':'region',

             'Happiness Rank':'rank',

             'Happiness.Rank':'rank',

             'Overall rank':'rank',

             'Happiness Score':'score',

             'Happiness.Score':'score',

             'Score':'score',

             'Economy (GDP per Capita)':'gdp_per_capita',

             'Economy..GDP.per.Capita.':'gdp_per_capita',

             'GDP per capita':'gdp_per_capita',

             'Family':'family',

             'Freedom':'freedom',

             'Freedom to make life choices':'freedom',

             'Generosity':'generosity',

             'Health (Life Expectancy)':'life_expectancy',

             'Health..Life.Expectancy.':'life_expectancy',

             'Healthy life expectancy':'life_expectancy',

             'Perceptions of corruption':'trust_corruption',

             'Trust (Government Corruption)':'trust_corruption',

             'Trust..Government.Corruption.':'trust_corruption',

             'Social support':'social_support',

             'Dystopia Residual':'dystopia_residual',

             'Dystopia.Residual':'dystopia_residual',

             'Standard Error':'standard_error',

             'Upper Confidence Interval':'whisker_high',

             'Whisker.high':'whisker_high',

             'Lower Confidence Interval':'whisker_low',

             'Whisker.low':'whisker_low'

            }



# Rename the columns

happiness.rename(columns=cols_dict, inplace=True)



print(happiness.columns) # check the new column names

happiness.head() # check the values
happiness.info()
# Duplicated

print('Duplicated: {}'.format(happiness.duplicated(subset='country').sum()))
cia_files = glob('/kaggle/input/the-world-factbook-by-cia/cia.health.*.201?.txt')

cia = pd.DataFrame()



for file in cia_files:

    c = pd.read_csv(file,

                    engine='python', sep=r'\s{3,}', header=None, names=['country_cia', file.split('.')[2]],

                    squeeze=False, skiprows=0, index_col=[0],

                    thousands=',', dtype={file.split('.')[2]:'float64'}

                   ) # read the file

    if cia.size == 0:

        cia = cia.append(c)

        print('Initialize: {}'.format(file.split('.')[2], cia.shape[0])) # for the first file

    else:

        cia = cia.merge(c, on='country_cia', how='outer')

        print('Merge {}: {}'.format(file.split('.')[2], cia.shape[0]))



cia.reset_index()



cia.info()

cia
cia['country'] = cia['country_cia']
happiness_cia = happiness.merge(cia, on='country', how='outer')[['country', 'score', 'infant_mortality']]



pd.set_option('display.max_rows', 100) # increase the number of rows to display

happiness_cia[happiness_cia.isnull().any(axis=1)].sort_values(by=['score', 'country']) # the countries don't match
# Countries map to rename

country_to_rename = {'Cote d\'Ivoire':'Ivory Coast',

                     'Congo, Republic of the':'Congo (Brazzaville)',

                     'Congo, Democratic Republic of the':'Congo (Kinshasa)',

                     'Burma':'Myanmar',

                     'Korea, South':'South Korea',

                     'Czechia':'Czech Republic'

                    }

# Rename the countries

cia['country'].replace(country_to_rename, inplace=True)



cia.sample(5, random_state=5) # check the values randomly
happiness_cia = happiness.merge(cia, on='country', how='left').copy()



happiness_cia.info()
# Select the columns of interest

cols_corr = ['country', 'region', 'score',

             'obesity', 'tfr',

             'mmr', 'infant_mortality',

             'hiv_aids', 'hiv_aids_death'

            ]

happiness_cia = happiness_cia[cols_corr]

happiness_cia
# Get correlation matrix

happiness_cia_corr = happiness_cia.corr()

happiness_cia_corr
# A triangular mask to avoid repeated values

happiness_cia_corr = happiness_cia_corr.iloc[1:, :-1]

mask = np.triu(np.ones_like(happiness_cia_corr), k=1)



# Readable names for the plot

cols_dict = {'score':'Happiness',

             'hiv_aids':'HIV/AIDS',

             'hiv_aids_death':'HIV/AIDS\ndeath',

             'infant_mortality':'Infant\nmortality',

             'mmr':'Maternal\nmortality',

             'tfr':'Total\nfertility',

             'obesity':'Obesity'

            }

# Rename columns in the correlation matrix

happiness_cia_corr.rename(columns=cols_dict, index=cols_dict, inplace=True)
%matplotlib inline



# Turn on svg rendering

%config InlineBackend.figure_format = 'svg'



# Color palette for the blog

snark_palette = ['#e0675a', # red

                 '#5ca0af', # green

                 '#edde7e', # yellow

                 '#211c47' # dark blue

                ]
# Color palette for the data

palette = [snark_palette[0], # red

           'lightgrey',

           snark_palette[1] # green

          ]



# Inscriptions

title = """Relationship Between Health Indicators And The Happiness score"""

description = """

Сorrelation of health indicators with the happiness score by country based on 2016 data.

Data: Gallup World Poll - www.kaggle.com/unsdsn/world-happiness & CIA - www.cia.gov/library/publications/the-world-factbook | Author: @data.sugar

"""



# Plot size

figsize = (6,4)



# Set the figure

sns.set(context='paper', style='ticks', palette=palette,

        rc={'xtick.bottom':False, 'ytick.left':False, 

            'axes.spines.left': False, 'axes.spines.bottom': False,

            'axes.spines.right': False, 'axes.spines.top': False

           }

       )



# Create the plot

fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='w')

sns.heatmap(happiness_cia_corr, mask=mask, cmap=palette,

            vmin=-1, vmax=1, center=0,

            square=False, linewidths=.5, annot=True, fmt='.2g',

            cbar_kws={'shrink': 1, 'ticks':[], 'label':'-1 negative <- correlation -> positive +1'},

            ax=ax)



# Set some aesthetic params for the plot

ax.set_title(title, fontdict={'fontsize': 16}, loc='center', pad=10, c=snark_palette[-1]) # set a title of the plot

ax.annotate(description, xy=(20, -4), size=6, xycoords='figure points', c=snark_palette[-1])

ax.tick_params(axis='both', colors=snark_palette[-1]) # set x/y ticks



# Save and plot

plt.savefig('/kaggle/working/plot.happiness.health.png', dpi=150, bbox_inches='tight')

plt.show()
# Inscriptions

title = """The Relationship Between Obesity And Happiness"""

description = """

Correlation of the obesity rate with the happiness score by country based on 2016 data.

Data: Gallup World Poll - www.kaggle.com/unsdsn/world-happiness & CIA - www.cia.gov/library/publications/the-world-factbook | Author: @data.sugar

"""



# Plot size

figsize = (6,4)



# Set the figure

sns.set(context='paper', style='ticks', palette=snark_palette,

        rc={'xtick.major.size': 4, 'ytick.major.size':4,

            'axes.spines.left': False, 'axes.spines.bottom': False,

            'axes.spines.right': False, 'axes.spines.top': False

           }

       )



# Create the plot

fig = plt.figure(figsize=figsize, facecolor='w')

g = sns.jointplot(x='score', y='obesity', data=happiness_cia,

                  kind='reg', truncate=False, dropna=True,

                  xlim=(2, 8), ylim=(0, 40),

                  marginal_kws=dict(hist=True, bins=10),

                  color=snark_palette[1]

                 )



# Set some aesthetic params for the plot

g.ax_marg_x.set_title(title, fontdict={'fontsize': 16}, loc='center', pad=10, c=snark_palette[-1]) # set a title of the plot

g.ax_marg_x.annotate(description, xy=(0.015, -0.01), size=6, xycoords='figure fraction', c=snark_palette[-1])

g.ax_joint.set_xlabel('Happiness score', horizontalalignment='center', size='x-large', c=snark_palette[-1]) # set label of x axis

g.ax_joint.set_ylabel('Obesity rate', horizontalalignment='center', size='x-large', c=snark_palette[-1]) # set label of x axis

g.ax_joint.tick_params(axis='both', labelsize='large', colors=snark_palette[-1]) # set x/y ticks

g.ax_joint.spines['bottom'].set_color(snark_palette[-1]) # color x axis

g.ax_joint.spines['left'].set_color(snark_palette[-1]) # color y axis

g.ax_marg_x.tick_params(axis='x', bottom=False) # disable x margin ticks

g.ax_marg_x.spines['bottom'].set_color(snark_palette[1])

g.ax_marg_y.tick_params(axis='y', left=False) # disable y margin ticks

g.ax_marg_y.spines['left'].set_color(snark_palette[1])



# Save and plot

plt.savefig('/kaggle/working/plot.happiness.health.obesity.png', dpi=150, bbox_inches='tight')

plt.show()
# Inscriptions

title = """The Relationship Between Total Fertility And Happiness"""

description = """

Correlation of the total fertility rate with the happiness score by country based on 2016 data.

Data: Gallup World Poll - www.kaggle.com/unsdsn/world-happiness & CIA - www.cia.gov/library/publications/the-world-factbook | Author: @data.sugar

"""



# Plot size

figsize = (6,4)



# Set the figure

sns.set(context='paper', style='ticks', palette=snark_palette,

        rc={'xtick.major.size': 4, 'ytick.major.size':4,

            'axes.spines.left': False, 'axes.spines.bottom': False,

            'axes.spines.right': False, 'axes.spines.top': False

           }

       )



# Create the plot

fig = plt.figure(figsize=figsize, facecolor='w')

g = sns.jointplot(x='score', y='tfr', data=happiness_cia,

                  kind='reg', truncate=False, dropna=True,

                  xlim=(2, 8), ylim=(0, 7),

                  marginal_kws=dict(hist=True, bins=8),

                  color=snark_palette[0]

                 )



# Set some aesthetic params for the plot

g.ax_marg_x.set_title(title, fontdict={'fontsize': 16}, loc='center', pad=10, c=snark_palette[-1]) # set a title of the plot

g.ax_marg_x.annotate(description, xy=(0.015, -0.01), size=6, xycoords='figure fraction', c=snark_palette[-1])

g.ax_joint.set_xlabel('Happiness score', horizontalalignment='center', size='x-large', c=snark_palette[-1]) # set label of x axis

g.ax_joint.set_ylabel('Total fertility rate', horizontalalignment='center', size='x-large', c=snark_palette[-1]) # set label of x axis

g.ax_joint.tick_params(axis='both', labelsize='large', colors=snark_palette[-1]) # set x/y ticks

g.ax_joint.spines['bottom'].set_color(snark_palette[-1]) # color x axis

g.ax_joint.spines['left'].set_color(snark_palette[-1]) # color y axis

g.ax_marg_x.tick_params(axis='x', bottom=False) # disable x margin ticks

g.ax_marg_x.spines['bottom'].set_color(snark_palette[0])

g.ax_marg_y.tick_params(axis='y', left=False) # disable y margin ticks

g.ax_marg_y.spines['left'].set_color(snark_palette[0])



# Save and plot

plt.savefig('/kaggle/working/plot.happiness.health.tfr.png', dpi=150, bbox_inches='tight')

plt.show()
# Inscriptions

title = """The Relationship Between Infant Mortality And Happiness"""

description = """

Correlation of the infant mortality rate with the happiness score by country based on 2016 data.

Data: Gallup World Poll - www.kaggle.com/unsdsn/world-happiness & CIA - www.cia.gov/library/publications/the-world-factbook | Author: @data.sugar

"""



# Plot size

figsize = (6,4)



# Set the figure

sns.set(context='paper', style='ticks', palette=snark_palette,

        rc={'xtick.major.size': 4, 'ytick.major.size':4,

            'axes.spines.left': False, 'axes.spines.bottom': False,

            'axes.spines.right': False, 'axes.spines.top': False

           }

       )



# Create the plot

fig = plt.figure(figsize=figsize, facecolor='w')

g = sns.jointplot(x='score', y='infant_mortality', data=happiness_cia,

                  kind='reg', truncate=False, dropna=True,

                  xlim=(2, 8), ylim=(0, 125),

                  marginal_kws=dict(hist=True, bins=10),

                  color=snark_palette[0]

                 )



# Set some aesthetic params for the plot

g.ax_marg_x.set_title(title, fontdict={'fontsize': 16}, loc='center', pad=10, c=snark_palette[-1]) # set a title of the plot

g.ax_marg_x.annotate(description, xy=(0.015, -0.01), size=6, xycoords='figure fraction', c=snark_palette[-1])

g.ax_joint.set_xlabel('Happiness score', horizontalalignment='center', size='x-large', c=snark_palette[-1]) # set label of x axis

g.ax_joint.set_ylabel('Infant mortality rate', horizontalalignment='center', size='x-large', c=snark_palette[-1]) # set label of x axis

g.ax_joint.tick_params(axis='both', labelsize='large', colors=snark_palette[-1]) # set x/y ticks

g.ax_joint.spines['bottom'].set_color(snark_palette[-1]) # color x axis

g.ax_joint.spines['left'].set_color(snark_palette[-1]) # color y axis

g.ax_marg_x.tick_params(axis='x', bottom=False) # disable x margin ticks

g.ax_marg_x.spines['bottom'].set_color(snark_palette[0])

g.ax_marg_y.tick_params(axis='y', left=False) # disable y margin ticks

g.ax_marg_y.spines['left'].set_color(snark_palette[0])



# Save and plot

plt.savefig('/kaggle/working/plot.happiness.health.infant_mortality.png', dpi=150, bbox_inches='tight')

plt.show()
# Inscriptions

title = """The Relationship Between Total Fertility And Obesity"""

description = """

Correlation of the total fertility rate with the obesity rate by country based on 2016 data.

Data: Gallup World Poll - www.kaggle.com/unsdsn/world-happiness & CIA - www.cia.gov/library/publications/the-world-factbook | Author: @data.sugar

"""



# Plot size

figsize = (6,4)



# Set the figure

sns.set(context='paper', style='ticks', palette=snark_palette,

        rc={'xtick.major.size': 4, 'ytick.major.size':4,

            'axes.spines.left': False, 'axes.spines.bottom': False,

            'axes.spines.right': False, 'axes.spines.top': False

           }

       )



# Create the plot

fig = plt.figure(figsize=figsize, facecolor='w')

g = sns.jointplot(x='obesity', y='tfr', data=happiness_cia,

                  kind='reg', truncate=False, dropna=True,

                  xlim=(0, 40), ylim=(0, 7),

                  marginal_kws=dict(hist=True, bins=10),

                  color=snark_palette[0]

                 )



# Set some aesthetic params for the plot

g.ax_marg_x.set_title(title, fontdict={'fontsize': 16}, loc='center', pad=10, c=snark_palette[-1]) # set a title of the plot

g.ax_marg_x.annotate(description, xy=(0.015, -0.01), size=6, xycoords='figure fraction', c=snark_palette[-1])

g.ax_joint.set_xlabel('Obesity rate', horizontalalignment='center', size='x-large', c=snark_palette[-1]) # set label of x axis

g.ax_joint.set_ylabel('Total fertility rate', horizontalalignment='center', size='x-large', c=snark_palette[-1]) # set label of x axis

g.ax_joint.tick_params(axis='both', labelsize='large', colors=snark_palette[-1]) # set x/y ticks

g.ax_joint.spines['bottom'].set_color(snark_palette[-1]) # color x axis

g.ax_joint.spines['left'].set_color(snark_palette[-1]) # color y axis

g.ax_marg_x.tick_params(axis='x', bottom=False) # disable x margin ticks

g.ax_marg_x.spines['bottom'].set_color(snark_palette[0])

g.ax_marg_y.tick_params(axis='y', left=False) # disable y margin ticks

g.ax_marg_y.spines['left'].set_color(snark_palette[0])



# Save and plot

plt.savefig('/kaggle/working/plot.happiness.health.tfr_obesity.png', dpi=150, bbox_inches='tight')

plt.show()