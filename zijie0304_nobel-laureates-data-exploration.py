import pandas as pd

import seaborn as sns

import numpy as np

nobel = pd.read_csv('../input/nobel-laureates/archive.csv')

nobel.head(n=6)
display(len(nobel))

display(nobel['Sex'].value_counts())

nobel['Birth Country'].value_counts().head(10)

nobel['usa_born_winner'] = nobel['Birth Country'] == 'United States of America'



nobel['decade'] = (np.floor(nobel['Year'] / 10) * 10).astype(int)



prop_usa_winners = nobel.groupby('decade', as_index=False)['usa_born_winner'].mean()

prop_usa_winners
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter



sns.set()

plt.rcParams['figure.figsize'] = [11, 7]

ax = sns.lineplot(x='decade', y='usa_born_winner', data=prop_usa_winners)

ax.yaxis.set_major_formatter(PercentFormatter(1.0))
# Calculating the proportion of female laureates per decade

nobel['female_winner'] = nobel['Sex'] == 'Female'

prop_female_winners = nobel.groupby(['decade', 'Category'], as_index=False)['female_winner'].mean()



# Plotting USA born winners with % winners on the y-axis

ax = sns.lineplot(x='decade', y='female_winner', hue='Category', data=prop_female_winners)

ax.yaxis.set_major_formatter(PercentFormatter(1.0))
nobel[nobel.Sex == 'Female'].nsmallest(1, 'Year')
nobel.groupby('Full Name').filter(lambda group: len(group) >= 2)
nobel['Birth Date'] = pd.to_datetime(nobel['Birth Date'],errors='coerce')



nobel['Age'] = nobel['Year'] - nobel['Birth Date'].dt.year

sns.lmplot(x='Year', y='Age', data=nobel, lowess=True, 

           aspect=2, line_kws={'Color' : 'black'})

sns.lmplot(x='Year', y='Age', row='Category', data=nobel, lowess=True, 

           aspect=2, line_kws={'color' : 'black'})

display(nobel.nlargest(1, 'Age'))



nobel.nsmallest(1, 'Age')