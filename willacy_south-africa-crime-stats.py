import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/SouthAfricaCrimeStats.csv')
df.head(3)
titles = list(df.columns)

dates = titles[7:17]

province = list(df.Province.value_counts().index)
plt.figure(figsize=(8,8))



# Plot each province's crimes onto the same graph.

for i in range(len(province)):

    df[df['Province'] == province[i]][dates].sum().plot(kind='line', linewidth=4)



plt.xticks(rotation=45)

plt.legend((province), loc='upper left')

plt.title('Crimes commited for each Province, per year.')

plt.ylabel('Crimes')

df[dates].sum().plot(kind='line')

plt.xticks(rotation=45)

plt.title('Overall number of crimes commited for each year')
# Population data taken from https://en.wikipedia.org/wiki/List_of_South_African_provinces_by_population

# Figures are for mid-2015. Only really comparable to the figures in column 2014-2015. 

province_population = {

                        'Gauteng':       13200300,

                        'Kwazulu/Natal': 10919100,

                        'Eastern Cape':   6916200,

                        'Western Cape':   6200100,

                        'Limpopo':        5726800,

                        'Mpumalanga':     4283900,

                        'North West':     3707100,

                        'Free State':     2817900,

                        'Northern Cape':  1185600

                      }
crimes_2015_df = pd.DataFrame(df.groupby('Province')['2014-2015'].sum())

crimes_2015_df.reset_index(inplace=True)

crimes_2015_df['Population'] = crimes_2015_df['Province'].map(province_population)

crimes_2015_df['Crime Ratio'] = crimes_2015_df['2014-2015']/crimes_2015_df['Population']
crimes_2015_df[['Crime Ratio','Province']].plot(kind='bar', x='Province')

plt.xticks(rotation=45)

plt.xlim(-1,len(crimes_2015_df.Province))

plt.title('Fraction of crimes commited per populous or province')
# Plotting line graphs for a select few types of crimes to get a sense of how they have changed over the course of ten years.

fig = plt.figure(figsize=(12,6), dpi=1600)

ax1 = plt.subplot2grid((2,3),(0,0))



df[df['Crime Category'] == 'Rape'][dates].sum().plot(kind='line')

plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

plt.xlabel('2005-2015')

plt.title('Rape')



ax2 = plt.subplot2grid((2,3),(0,1))

df[df['Crime Category'] == 'Drug-related crime'][dates].sum().plot(kind='line')

plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

plt.xlabel('2005-2015')

plt.title('Drug-related Crime')



ax3 = plt.subplot2grid((2,3),(0,2))

df[df['Crime Category'] == 'Murder'][dates].sum().plot(kind='line')

plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

plt.xlabel('2005-2015')

plt.title('Murder')



ax4 = plt.subplot2grid((2,3),(1,0))

df[df['Crime Category'] == 'Arson'][dates].sum().plot(kind='line')

plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

plt.xlabel('2005-2015')

plt.title('Arson')



ax5 = plt.subplot2grid((2,3),(1,1))

df[df['Crime Category'] == 'Abduction'][dates].sum().plot(kind='line')

plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

plt.xlabel('2005-2015')

plt.title('Abduction')



ax6 = plt.subplot2grid((2,3),(1,2))

df[df['Crime Category'] == 'Neglect and ill-treatment of children'][dates].sum().plot(kind='line')

plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

plt.xlabel('2005-2015')

plt.title('Neglect and Ill-treatment of Children')
df[df['Crime Category'] == 'Arson'][dates].sum().plot(kind='bar', label='Arson', alpha=.55)

df[df['Crime Category'] == 'Neglect and ill-treatment of children'][dates].sum().plot(kind='bar', label='Child Neglect', color='#AABC45')

df[df['Crime Category'] == 'Abduction'][dates].sum().plot(kind='bar', color='#123456', label='Abduction')

plt.legend()