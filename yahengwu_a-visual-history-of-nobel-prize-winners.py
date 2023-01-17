# Loading in required libraries

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Reading in the Nobel Prize data

nobel = pd.read_csv("../input/nobel-laureates/archive.csv")
# Taking a look at the dataset

display(nobel.shape)

display(nobel.columns)

display(nobel.head())
# Display the number of prizes won by the top 10 nationalities from 1901 to 2016

nobel['Birth Country'].value_counts().head(n=10)
# Display the number of prizes won by sex from 1901 to 2016

sns.set(style='white')

ax = sns.countplot(x='Sex', palette=["#3498db", "#e74c3c"], data=nobel)

ax.set_title('Nobel prizes won by male and female between 1901 and 2016', fontsize=15)
# Display the number of nobel prizes given by category

sns.set(style='white')

ax = sns.countplot(x='Category', palette = 'GnBu_d', order = nobel['Category'].value_counts().index, data=nobel)

ax.set_title('Nobel prizes given in total by category from 1901 to 2016', fontsize=15)
# Calculating the proportion of nobel prize winners per decade

nobel['Decade'] = (np.floor(nobel['Year'] / 10) * 10).astype(int)



# Display the number of nobel prizes handed out per decade

sns.set(style='white')

ax = sns.countplot(x='Decade', palette = 'Blues', order = nobel['Decade'].value_counts(ascending=True).index, data=nobel)

ax.set_title('The number of nobel prize handed out per decade', fontsize=15)
# Calculating the proportion of female recipient per decade

nobel['female_winner'] = nobel['Sex']=='Female'

prop_female_winners = nobel.groupby(['Decade','Category'],as_index=False)['female_winner'].mean()

ax = sns.lineplot(x='Decade', y='female_winner',label="Female", color = '#e74c3c', data=prop_female_winners)





# Calculating the proportion of male recipient per decade

nobel['male_winner'] = nobel['Sex']=='Male'

prop_male_winners = nobel.groupby(['Decade','Category'],as_index=False)['male_winner'].mean()

ax = sns.lineplot(x='Decade', y='male_winner',label="Male", data=prop_male_winners)



from matplotlib.ticker import PercentFormatter

# Adding %-formatting to the y-axis

ax.yaxis.set_major_formatter(PercentFormatter(1.0))

ax.set_title('Nobel prize winners proportion by sex from 1901 to 2016', fontsize=15)