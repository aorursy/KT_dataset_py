# basic operations

import numpy as np

import pandas as pd 



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# file path

import os

print(os.listdir("../input/fifa-19"))
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%time data = pd.read_csv('/kaggle/input/fifa-19/data.csv')



print(data.shape)
data.head()
def club(x):

    return data[data['Club'] == x][['Name','Jersey Number','Position','Overall','Nationality','Age','Wage',

                                    'Value','Contract Valid Until']]



club('Manchester United')
plt.rcParams['figure.figsize'] = (10, 5)

sns.countplot(data['Preferred Foot'], palette = 'pink')

plt.title('Most Preferred Foot of the Players', fontsize = 20)

plt.show()
plt.figure(figsize = (18, 8))

plt.style.use('fivethirtyeight')

ax = sns.countplot('Position', data = data, palette = 'bone')

ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)

ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)

plt.show()


def extract_value_from(Value):

    out = Value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in Value:

        out = float(out.replace('K', ''))*1000

    return float(out)

# applying the function to the wage column



data['Value'] = data['Value'].apply(lambda x: extract_value_from(x))

data['Wage'] = data['Wage'].apply(lambda x: extract_value_from(x))



data['Wage'].head()

import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (15, 5)

sns.distplot(data['Wage'], color = 'blue')

plt.xlabel('Wage Range for Players', fontsize = 16)

plt.ylabel('Count of the Players', fontsize = 16)

plt.title('Distribution of Wages of Players', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (10, 8))

ax = sns.countplot(x = 'Skill Moves', data = data, palette = 'pastel')

ax.set_title(label = 'Count of players on Basis of their skill moves', fontsize = 20)

ax.set_xlabel(xlabel = 'Number of Skill Moves', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'Height', data = data, palette = 'dark')

ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)

ax.set_xlabel(xlabel = 'Height in Foot per inch', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()


plt.style.use('dark_background')

data['Nationality'].value_counts().head(80).plot.bar(color = 'orange', figsize = (20, 7))

plt.title('Different Nations Participating in FIFA 2019', fontsize = 30, fontweight = 20)

plt.xlabel('Name of The Country')

plt.ylabel('count')

plt.show()
sns.set(style = "dark", palette = "colorblind", color_codes = True)

x = data.Age

plt.figure(figsize = (15,8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'g')

ax.set_xlabel(xlabel = "Player\'s age", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Histogram of players age', fontsize = 20)

plt.show()