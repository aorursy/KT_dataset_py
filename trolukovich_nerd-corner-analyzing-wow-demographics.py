# Loading all necessary modules

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

import cufflinks



cufflinks.go_offline(connected=True)

init_notebook_mode(connected = True)

%matplotlib inline



# To see more columns

pd.set_option('display.max_columns', 100)
# loading dataset

data = pd.read_csv('../input/WoW Demographics.csv')

data.head()
data = data.drop('Timestamp', axis = 1)

data.head()
# Shape of dataset

data.shape
data.info()
# Lets take a quicklook at unique values

for col in data.columns.values[1:]:

    print(col, ': ' ,data[col].unique())
# Let's start with a simple

data['Country'] = data['Country'].str.capitalize()

data['Country'] = data['Country'].replace('Uk', 'U.k')

data[['Gender', 'Sexuality', 'Age', 'Country']].iplot(kind = 'hist', 

                                                      yTitle = 'Count', 

                                                      title = 'Variable distribution',

                                                      subplots = True, 

                                                      shape = (2, 2))
def stat(col, z=2.58):

    n = len(data[col])

    

    for i in range(len(data[col].value_counts().index.values)):

        p = data[col].value_counts()[i]/n

        pred = z*np.sqrt((p*(1-p))/n)

        print(data[col].value_counts().index[i],': {0:.2f}% +-{1:.2f}'.format(p*100, pred*100))



stat('Gender')
# Same thing for Sexuality and Age

stat('Sexuality')
stat('Age')
# Let's dig a little deeper

fig, ax = plt.subplots(1, 3, figsize = (18, 5))

sns.countplot('Sexuality', data = data, hue = 'Gender', ax = ax[0])

sns.countplot('Age', data = data, hue = 'Gender', ax = ax[1])

sns.countplot('Age', data = data, hue = 'Sexuality', ax = ax[2])
data[['Main', 'Faction', 'Max']].iplot(kind = 'hist', 

                                       yTitle = 'Count', 

                                       title = 'Main\Faction\Max distribution',

                                       subplots = True, 

                                       shape = (1, 3))
fig, ax = plt.subplots(1, 3, figsize = (18, 4))

sns.countplot('Main', data = data, hue = 'Gender', ax = ax[0])

sns.countplot('Faction', data = data, hue = 'Gender', ax = ax[1])

sns.countplot('Max', data = data, hue = 'Gender', ax = ax[2])
# Let's look at our most interesting columns

# First - drop Na values

na = data.loc[data['Server'].isna()].index

data = data.drop(na, axis = 0)

data.info()
# I want to create dummy variables for every value

def dummies(cols, target):

    for col in cols:

        data[col] = 0

        data.loc[data[target].str.contains(col, regex = False), col] = 1

    data.drop(target, axis = 1, inplace = True)

        

cols = ['PvE', 'PvP', 'RP']

dummies(cols, 'Server')



role = ['DPS', 'Healer', 'Tank']

dummies(role, 'Role')



clas = ['Hunter', 'Druid', 'Priest', 'Shaman', 'Death Knight', 'Demon Hunter', 'Paladin', 'Warlock', 'Warrior',

       'Monk', 'Rogue', 'Mage']

dummies(clas, 'Class')



race = ['Draenei', 'Troll', 'Night Elf', 'Dwarf', 'Blood Elf', 'Tauren', 'Pandaren', 'Gnome', 'Human',

       'Orc', 'Goblin', 'Undead', 'Worgen']

dummies(race, 'Race')
# Let's see what we got

data.head()
s = data[cols + role + clas + race].sum()

fig, ax = plt.subplots(2, 2, figsize = (18, 8))

s[cols].plot(kind = 'bar', ax = ax[0, 0])

s[role].plot(kind = 'bar', ax = ax[0, 1])

s[clas].plot(kind = 'bar', ax = ax[1, 0])

s[race].plot(kind = 'bar', ax = ax[1, 1])
serv = data.pivot_table(cols + role + clas + race, ['Gender'], aggfunc = 'sum')



x = ['PvE', 'PvP', 'RP']

y1 = list(serv.loc['Female', ['PvE', 'PvP', 'RP']].values)

y2 = list(serv.loc['Male', ['PvE', 'PvP', 'RP']].values)

y3 = list(serv.loc['Other', ['PvE', 'PvP', 'RP']].values)



trace1 = go.Bar(x = x, 

               y = y1,

               name = 'Female')



trace2 = go.Bar(x = x, 

               y = y2,

               name = 'Male')



trace3 = go.Bar(x = x, 

               y = y3,

               name = 'Other')



dt = [trace1, trace2, trace3]

layout = go.Layout(title = 'Server distribution by gender')

fig = go.Figure(data = dt, layout = layout)

fig.iplot()
x = role

y1 = list(serv.loc['Female', role].values)

y2 = list(serv.loc['Male', role].values)

y3 = list(serv.loc['Other', role].values)



trace1 = go.Bar(x = x, 

               y = y1,

               name = 'Female')



trace2 = go.Bar(x = x, 

               y = y2,

               name = 'Male')



trace3 = go.Bar(x = x, 

               y = y3,

               name = 'Other')



dt = [trace1, trace2, trace3]

layout = go.Layout(title = 'Role distribution by gender')

fig = go.Figure(data = dt, layout = layout)

fig.iplot()
x = clas

y1 = list(serv.loc['Female', clas].values)

y2 = list(serv.loc['Male', clas].values)

y3 = list(serv.loc['Other', clas].values)



trace1 = go.Bar(x = x, 

               y = y1,

               name = 'Female')



trace2 = go.Bar(x = x, 

               y = y2,

               name = 'Male')



trace3 = go.Bar(x = x, 

               y = y3,

               name = 'Other')



dt = [trace1, trace2, trace3]

layout = go.Layout(title = 'Class distribution by gender')

fig = go.Figure(data = dt, layout = layout)

fig.iplot()
x = race

y1 = list(serv.loc['Female', race].values)

y2 = list(serv.loc['Male', race].values)

y3 = list(serv.loc['Other', race].values)



trace1 = go.Bar(x = x, 

               y = y1,

               name = 'Female')



trace2 = go.Bar(x = x, 

               y = y2,

               name = 'Male')



trace3 = go.Bar(x = x, 

               y = y3,

               name = 'Other')



dt = [trace1, trace2, trace3]

layout = go.Layout(title = 'Race distribution by gender')

fig = go.Figure(data = dt, layout = layout)

fig.iplot()