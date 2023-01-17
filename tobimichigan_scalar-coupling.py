# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os







#dir_folder="input/Predicting Molecular Properties"

# Read the file containing the molecular structures

df_structures = pd.read_csv("../input/champs-scalar-coupling/structures.csv")

df_structures.head()



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/champs-scalar-coupling/train.csv")

df_train.head()
df_test=pd.read_csv("../input/champs-scalar-coupling/test.csv")

df_test.head()
df_dipole_moments=pd.read_csv("../input/champs-scalar-coupling/dipole_moments.csv")

df_dipole_moments.head()
def combine_molecule_structure(data: pd.DataFrame):

    atom_idx=[0,1]

    for idx in atom_idx:

        data = data.merge( df_structures, how='left',

                  left_on = ['molecule_name', f'atom_index_{idx}'],

                  right_on = ['molecule_name', 'atom_index'])

        

        data= data.drop('atom_index', axis=1)

        data=data.rename(columns={'atom': f'atom_{idx}',

                                    'x': f'x_{idx}',

                                    'y': f'y_{idx}',

                                    'z': f'z_{idx}'})

        return data

                   
df_train=combine_molecule_structure(df_train)

df_train.head
