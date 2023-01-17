!pip install pandas-profiling
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pandas_profiling # fast EDA tool



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import os

print(os.listdir("../input"))
from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('../input/train.csv', index_col='id')

test = pd.read_csv('../input/test.csv', index_col='id')
display(train.head())
structures = pd.read_csv('../input/structures.csv')

display(structures.head())
# Map the atom structure data into train and test files



def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)
train.head()
test.head()
pandas_profiling.ProfileReport(train)
pandas_profiling.ProfileReport(test)