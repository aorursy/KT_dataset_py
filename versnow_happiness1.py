import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output



data2015 = pd.read_csv('../input/2015.csv')

data2016 = pd.read_csv('../input/2016.csv')

data2017 = pd.read_csv('../input/2017.csv')



print(data2017.columns)



# Correlation heatmap

corr = data2017[['Happiness.Score', 'Economy..GDP.per.Capita.', 'Family',

       'Health..Life.Expectancy.', 'Freedom', 'Generosity',

       'Trust..Government.Corruption.']].corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(corr, annot=True, fmt= '.1f', linewidths=.5, center=1, cbar=False, mask=mask)
# Scatter plot between Happiness score and Economy GDP/Capita

sns.regplot(x='Economy (GDP per Capita)', y='Happiness Score', data=data2015);
g = sns.stripplot(x="Region", y="Happiness Score", data=data2015, jitter=True)

plt.xticks(rotation=90)
g = sns.stripplot(x="Region", y="Economy (GDP per Capita)", data=data2015, jitter=True)

plt.xticks(rotation=90)