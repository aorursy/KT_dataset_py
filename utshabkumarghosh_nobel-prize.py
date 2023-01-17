# Loading in required libraries

import pandas as pd

import seaborn as sns

import numpy as np



# Reading in the Nobel Prize data

nobel = pd.read_csv('../input/nobel-prize-19012016/nobel.csv')



# Taking a look at the first several winners

nobel.head(6)
# Display the number of (possibly shared) Nobel Prizes handed

# out between 1901 and 2016

display(len(nobel))



# Display the number of prizes won by male and female recipients.

display(nobel['sex'].value_counts())



# Display the number of prizes won by the top 10 nationalities.

nobel['birth_country'].value_counts().head(10)
# Calculating the proportion of USA born winners per decade

nobel['usa_born_winner'] = nobel['birth_country']=='United States of America'

nobel['decade'] = (np.floor(nobel['year'] / 10) * 10).astype(int)

prop_usa_winners = nobel.groupby('decade', as_index=False)['usa_born_winner'].mean()



# Display the proportions of USA born winners per decade

display(prop_usa_winners)
# Setting the plotting theme

sns.set()

# and setting the size of all plots.

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [11, 7]



# Plotting USA born winners 

ax = sns.lineplot(x='decade', y='usa_born_winner', data=nobel)



# Adding %-formatting to the y-axis

import matplotlib.ticker as mtick

from matplotlib.ticker import PercentFormatter

ax.yaxis.set_major_formatter(mtick.PercentFormatter())
# Calculating the proportion of female laureates per decade

nobel['female_winner'] = nobel['sex'] == 'Female'

prop_female_winners = nobel.groupby(['decade', 'category'], as_index=False)['female_winner'].mean()



# Plotting female winners with % winners on the y-axis

ax = sns.lineplot(x='decade', y='female_winner', hue='category', data=prop_female_winners)

ax.yaxis.set_major_formatter(PercentFormatter(1.0))
# Picking out the first woman to win a Nobel Prize

nobel[nobel.sex == 'Female'].nsmallest(1, 'year')
# Selecting the laureates that have received 2 or more prizes.

nobel.groupby('full_name').filter(lambda group: len(group) >= 2)
# Converting birth_date from String to datetime

nobel['birth_date'] = pd.to_datetime(nobel['birth_date'])



# Calculating the age of Nobel Prize winners

nobel['age'] = nobel['year'] - nobel['birth_date'].dt.year



# Plotting the age of Nobel Prize winners

sns.lmplot(data=nobel, y='age', x='year', lowess=True, aspect=2, line_kws={'color' : 'black'})
# Same plot as above, but separate plots for each type of Nobel Prize

sns.lmplot(x='year', y='age', row='category', data=nobel)
# The oldest winner of a Nobel Prize as of 2016

display(nobel[nobel.year==2016].nsmallest(1, 'age'))



# The youngest winner of a Nobel Prize as of 2016

display(nobel[nobel.year==2016].nlargest(1, 'age'))
# The name of the youngest winner of the Nobel Prize as of 2016

youngest_winner = '...' ## Google it :3