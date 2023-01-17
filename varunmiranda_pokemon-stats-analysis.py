import pandas as pd

import numpy as np

import seaborn as sns

import scipy as sp

import sklearn



from matplotlib import pyplot as plt

from scipy.stats import norm, skew

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import neighbors

from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import neural_network

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

import datetime

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', 500)

import warnings

warnings.filterwarnings('ignore')
dataset = pd.read_csv('../input/Pokemon.csv')

dataset.head()
duplicate_nums = pd.DataFrame(dataset['#'].value_counts())

duplicate_nums = duplicate_nums[duplicate_nums['#'] > 1]

duplicate_nums.reset_index(level=0, inplace=True)

duplicate_entries = dataset[dataset['#'].isin(duplicate_nums['index'])]

mega_evolution = duplicate_entries[duplicate_entries['Name'].str.contains('Mega')]

mega_evolution
dataset = pd.merge(dataset, mega_evolution[['Name']], on=['Name'], how='left', indicator='Mega_Flag')

dataset['Mega_Flag'] = np.where(dataset.Mega_Flag == 'both', 1, 0)

dataset.head()
dataset.rename(columns={'Type 1': 'Type_1', 'Type 2': 'Type_2'}, inplace=True)

q1a = pd.value_counts(dataset.Type_1).to_frame().reset_index()

q1b = pd.value_counts(dataset.Type_2).to_frame().reset_index()

q1a.rename(columns={'Type_1': 'Type'}, inplace=True)

q1b.rename(columns={'Type_2': 'Type'}, inplace=True)

q1 = q1a.append(q1b)
#The value counts function doesn't account for NaNs

q1.fillna(0)
q1 = q1.groupby(['index']).sum()
q1.reset_index(level=0, inplace=True)

q1 = q1.sort_values(by=['Type'],ascending = False)
ax = sns.catplot(x="Type",y="index", data= q1)

ax.set(xlabel='Number of Pokemon', ylabel='Pokemon Type')

plt.show()
dataset['Dual_Type']  = np.where(dataset.Type_2.isnull(), "No", "Yes")

dataset.head(10)
q2 = pd.value_counts(dataset.Dual_Type).to_frame().reset_index()

q2.set_index('index')
q2.plot(kind = "pie", y="Dual_Type", autopct='%1.1f%%',labels=q2['index'])
q3a = dataset[['Type_1','Generation','Legendary','Mega_Flag']]

q3b = dataset[['Type_2','Generation','Legendary','Mega_Flag']]

q3a.rename(columns={'Type_1': 'Type'}, inplace=True)

q3b.rename(columns={'Type_2': 'Type'}, inplace=True)

q3 = q3a.append(q3b)

q3.head()
q3 = q3.groupby(['Type','Generation']).count()[['Legendary']].reset_index()

q3.head()
g = sns.FacetGrid(q3, col="Generation",height=4, aspect=4, col_wrap=1)

g = g.map(plt.bar,'Type','Legendary',color=['lightgreen', 'black', 'darkslateblue', 'yellow', 'pink'

                                            ,'brown','red','mediumpurple','indigo','limegreen','khaki',

                                           'lightcyan','lightgrey','purple','deeppink','darkgoldenrod',

                                           'lightslategrey','dodgerblue'])

g.add_legend()

plt.show()
q3a = dataset[['Type_1','Generation','Legendary','Mega_Flag']]

q3b = dataset[['Type_2','Generation','Legendary','Mega_Flag']]

q3a.rename(columns={'Type_1': 'Type'}, inplace=True)

q3b.rename(columns={'Type_2': 'Type'}, inplace=True)

q3 = q3a.append(q3b)

q3.head()
q3['Legendary'].value_counts()
q3 = q3.loc[q3.Legendary]
q3 = q3.groupby(['Type']).count()[['Legendary']].reset_index()

q3.head()
q3 = q3.sort_values(by=['Legendary'],ascending = False)

ax = sns.catplot(x="Legendary",y="Type", data= q3)

ax.set(xlabel='Number of Legendary Pokemon', ylabel='Pokemon Type')

plt.show()
q3a = dataset[['Type_1','Generation','Legendary','Mega_Flag']]

q3b = dataset[['Type_2','Generation','Legendary','Mega_Flag']]

q3a.rename(columns={'Type_1': 'Type'}, inplace=True)

q3b.rename(columns={'Type_2': 'Type'}, inplace=True)

q3 = q3a.append(q3b)

q3.head()
q3 = pd.DataFrame(q3.loc[q3['Mega_Flag'] == 1])

q3.head()
q3 = q3.groupby(['Type']).count()[['Legendary']].reset_index()

q3.head()
q3 = q3.sort_values(by=['Legendary'],ascending = False)

ax = sns.catplot(x="Legendary",y="Type", data= q3)

ax.set(xlabel='Number of Pokemon that can Mega Evolve', ylabel='Pokemon Type')

q3.head()
q4raw_1 = dataset[['Type_1','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]

q4raw_2 = dataset[['Type_2','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]

q4raw_1.rename(columns={'Type_1': 'Type'}, inplace=True)

q4raw_2.rename(columns={'Type_2': 'Type'}, inplace=True)

q4raw = q4raw_1.append(q4raw_2)

q4raw.rename(columns={'Sp. Atk': 'Special_Attack'}, inplace=True)

q4raw.rename(columns={'Sp. Def': 'Special_Defense'}, inplace=True)

q4raw.head()
q4raw = q4raw[q4raw.Type.notnull()]

q4raw.shape
q4melt = pd.melt(q4raw, id_vars = ['Type'], value_vars = ['HP', 'Attack','Defense','Special_Attack','Special_Defense','Speed'])

q4melt.head()
q4melt.rename(columns={'variable': 'Stat_Type'}, inplace=True)

q4melt.rename(columns={'value': 'Stat_Value'}, inplace=True)

sns.set()

g = sns.catplot(x = "Stat_Type", y = "Stat_Value", col = "Type", data = q4melt, kind = "violin", col_wrap = 1, aspect = 3)

plt.grid(True)