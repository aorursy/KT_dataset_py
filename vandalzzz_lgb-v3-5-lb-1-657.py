# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline



import pandas as pd

import numpy as np



import math

import gc

import copy



from sklearn.model_selection import GroupKFold, train_test_split

from sklearn.metrics import mean_absolute_error



import matplotlib.pyplot as plt

import seaborn as sns





from lightgbm import LGBMRegressor

import lightgbm as lgb
DATA_PATH = '/kaggle/input/champs-scalar-coupling/'

SUBMISSIONS_PATH = '.'

# use atomic numbers to recode atomic names

ATOMIC_NUMBERS = {

    'H': 1,

    'C': 6,

    'N': 7,

    'O': 8,

    'F': 9

}
pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', 120)

pd.set_option('display.max_columns', 120)
train_dtypes = {

    'molecule_name': 'category',

    'atom_index_0': 'int8',

    'atom_index_1': 'int8',

    'type': 'category',

    'scalar_coupling_constant': 'float32'

}

train_csv = pd.read_csv(f'{DATA_PATH}/train.csv', index_col='id', dtype=train_dtypes)

train_csv['molecule_index'] = train_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')

train_csv = train_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']]

train_csv.head(10)
print('Shape: ', train_csv.shape)

print('Total: ', train_csv.memory_usage().sum())

train_csv.memory_usage()
submission_csv = pd.read_csv(f'{DATA_PATH}/sample_submission.csv', index_col='id')
test_csv = pd.read_csv(f'{DATA_PATH}/test.csv', index_col='id', dtype=train_dtypes)

test_csv['molecule_index'] = test_csv['molecule_name'].str.replace('dsgdb9nsd_', '').astype('int32')

test_csv = test_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type']]

test_csv.head(10)
structures_dtypes = {

    'molecule_name': 'category',

    'atom_index': 'int8',

    'atom': 'category',

    'x': 'float32',

    'y': 'float32',

    'z': 'float32'

}

structures_csv = pd.read_csv(f'{DATA_PATH}/structures.csv', dtype=structures_dtypes)

structures_csv['molecule_index'] = structures_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')

structures_csv = structures_csv[['molecule_index', 'atom_index', 'atom', 'x', 'y', 'z']]

structures_csv['atom'] = structures_csv['atom'].replace(ATOMIC_NUMBERS).astype('int8')

structures_csv.head(10)
print('Shape: ', structures_csv.shape)

print('Total: ', structures_csv.memory_usage().sum())

structures_csv.memory_usage()
def build_type_dataframes(base, structures, coupling_type):

    base = base[base['type'] == coupling_type].drop('type', axis=1).copy()

    base = base.reset_index()

    base['id'] = base['id'].astype('int32')

    structures = structures[structures['molecule_index'].isin(base['molecule_index'])]

    return base, structures
def add_coordinates(base, structures, index):

    df = pd.merge(base, structures, how='inner',

                  left_on=['molecule_index', f'atom_index_{index}'],

                  right_on=['molecule_index', 'atom_index']).drop(['atom_index'], axis=1)

    df = df.rename(columns={

        'atom': f'atom_{index}',

        'x': f'x_{index}',

        'y': f'y_{index}',

        'z': f'z_{index}'

    })

    return df
def add_atoms(base, atoms):

    df = pd.merge(base, atoms, how='inner',

                  on=['molecule_index', 'atom_index_0', 'atom_index_1'])

    return df
def merge_all_atoms(base, structures):

    df = pd.merge(base, structures, how='left',

                  left_on=['molecule_index'],

                  right_on=['molecule_index'])

    df = df[(df.atom_index_0 != df.atom_index) & (df.atom_index_1 != df.atom_index)]

    return df
def add_center(df):

    df['x_c'] = ((df['x_1'] + df['x_0']) * np.float32(0.5))

    df['y_c'] = ((df['y_1'] + df['y_0']) * np.float32(0.5))

    df['z_c'] = ((df['z_1'] + df['z_0']) * np.float32(0.5))



#did find how to deal with data processing funtion at the moment of creation this, therefore another one center metween two atoms

def add_newcenter(df):

    df['centx'] = ((df['x_1'] + df['x_0']) * np.float32(0.5))

    df['centy'] = ((df['y_1'] + df['y_0']) * np.float32(0.5))

    df['centz'] = ((df['z_1'] + df['z_0']) * np.float32(0.5))



def add_distance_to_center(df):

    df['d_c'] = ((

        (df['x_c'] - df['x'])**np.float32(2) +

        (df['y_c'] - df['y'])**np.float32(2) + 

        (df['z_c'] - df['z'])**np.float32(2)

    )**np.float32(0.5))



def add_distance_between(df, suffix1, suffix2):

    df[f'd_{suffix1}_{suffix2}'] = ((

        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'] +1e-10)**np.float32(2) +

        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'] +1e-10)**np.float32(2) + 

        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'] +1e-10)**np.float32(2)

    )**np.float32(0.5))



# distance btw center of two target atoms and center of another non-target pair of atoms

def add_distance_ctoc(df, suffix1, suffix2):

    df[f'dc_{suffix1}_{suffix2}'] = ((

        ((df[f'x_{suffix1}'] + df[f'x_{suffix2}'])*np.float32(0.5) - df['centx'])**np.float32(2) +

        ((df[f'y_{suffix1}'] + df[f'y_{suffix2}'])*np.float32(0.5) - df['centy'])**np.float32(2) + 

        ((df[f'z_{suffix1}'] + df[f'z_{suffix2}'])*np.float32(0.5) - df['centz'])**np.float32(2)

    )**np.float32(0.5))

    

# cube distance is from some big book about nuclear or quantum physics

# should try to add same for target atom vs non-target atom

def add_cube_dist(df, suffix1, suffix2):

    df[f'd3_{suffix1}_{suffix2}'] = 1/((((

        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'] +1e-10)**np.float32(2) +

        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'] +1e-10)**np.float32(2) + 

        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'] +1e-10)**np.float32(2)

    )**np.float32(0.5)))**np.float32(3))
def add_distances(df):

    n_atoms = 1 + max([int(c.split('_')[1]) for c in df.columns if c.startswith('x_')])

    

    for i in range(1, n_atoms):

        for vi in range(min(4, i)):

            add_distance_between(df, i, vi)

            add_distance_ctoc(df, i, vi)

            add_cube_dist(df, i, vi)
def add_n_atoms(base, structures):

    dfs = structures['molecule_index'].value_counts().rename('n_atoms').to_frame()

    return pd.merge(base, dfs, left_on='molecule_index', right_index=True)
def build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=10):

    base, structures = build_type_dataframes(some_csv, structures_csv, coupling_type)

    base = add_coordinates(base, structures, 0)

    base = add_coordinates(base, structures, 1)

    

    base = base.drop(['atom_0', 'atom_1'], axis=1)

    

    atoms = base.drop('id', axis=1).copy()

    if 'scalar_coupling_constant' in some_csv:

        atoms = atoms.drop(['scalar_coupling_constant'], axis=1)

    add_newcenter(base)    

    add_center(atoms)



    atoms = atoms.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], axis=1)



    atoms = merge_all_atoms(atoms, structures)

    

    add_distance_to_center(atoms)

    

    atoms = atoms.drop(['x_c', 'y_c', 'z_c', 'atom_index'], axis=1)



    atoms.sort_values(['molecule_index', 'atom_index_0', 'atom_index_1', 'd_c'], inplace=True)

    atom_groups = atoms.groupby(['molecule_index', 'atom_index_0', 'atom_index_1'])

    atoms['num'] = atom_groups.cumcount() + 2

    atoms = atoms.drop(['d_c'], axis=1)

    atoms = atoms[atoms['num'] < n_atoms]



    atoms = atoms.set_index(['molecule_index', 'atom_index_0', 'atom_index_1', 'num']).unstack()

    atoms.columns = [f'{col[0]}_{col[1]}' for col in atoms.columns]

    atoms = atoms.reset_index()

    

    # downcast back to int8

    for col in atoms.columns:

        if col.startswith('atom_'):

            atoms[col] = atoms[col].fillna(0).astype('int8')

            

    atoms['molecule_index'] = atoms['molecule_index'].astype('int32')

    

    full = add_atoms(base, atoms)



    add_distances(full)



    full['numofatoms'] = n_atoms

    full['numofatoms'] = full['numofatoms'].astype('int8')

    full.sort_values('id', inplace=True)

    

    return full
def take_n_atoms(df, n_atoms, four_start=4):

    labels = []

    for i in range(2, n_atoms):

        label = f'atom_{i}'

        labels.append(label)



    for i in range(n_atoms):

        num = min(i, 4) if i < four_start else 4

        for j in range(num):

            labels.append(f'd_{i}_{j}')

            labels.append(f'd3_{i}_{j}')

            labels.append(f'dc_{i}_{j}')

    if 'scalar_coupling_constant' in df:

        labels.append('scalar_coupling_constant')

    return df[labels]
%%time

full = build_couple_dataframe(train_csv, structures_csv, '1JHN', n_atoms=10)

print(full.shape)
df = take_n_atoms(full, 7)

# LightGBM performs better with 0-s then with NaN-s

df = df.fillna(0)

df.columns
X_data = df.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')

y_data = df['scalar_coupling_constant'].values.astype('float32')



X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=128)

X_train.shape, X_val.shape, y_train.shape, y_val.shape

# lgb.Dataset(data=)
gc.collect()
LGB_PARAMS = {

    'objective': 'regression',

    'metric': 'mae',

    'verbosity': -1,

    'boosting_type': 'gbdt',

    'learning_rate': 0.07,

    'num_leaves': 256,

    'min_child_samples': 79,

    'max_depth': 18,

    'subsample_freq': 1,

    'subsample': 0.9,

    'bagging_seed': 11,

    'reg_alpha': 0.1,

    'reg_lambda': 0.3,

#     'colsample_bytree': 1.0 #v.3.1

    'colsample_bytree': 0.8 # v3.5

}
def build_x_y_data(some_csv, coupling_type, n_atoms):

    full = build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=n_atoms)

    

    df = take_n_atoms(full, n_atoms)

    df = df.fillna(0)

    print(df.columns)

    

    if 'scalar_coupling_constant' in df:

        X_data = df.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')

        y_data = df['scalar_coupling_constant'].values.astype('float32')

    else:

        X_data = df.values.astype('float32')

        y_data = None

    

    return X_data, y_data, full.molecule_index.values
def build_x_y_data(some_csv, coupling_type, n_atoms):

    full = build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=n_atoms)

    

    df = take_n_atoms(full, n_atoms)

    df = df.fillna(0)

    print(df.columns)

    

    if 'scalar_coupling_constant' in df:

        X_data = df.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')

        y_data = df['scalar_coupling_constant'].values.astype('float32')

    else:

        X_data = df.values.astype('float32')

        y_data = None

    

    return X_data, y_data, full.molecule_index.values
def train_and_predict_for_one_coupling_type(coupling_type, submission, n_atoms, n_folds=5, n_splits=5, random_state=42):

    print(f'*** Training Model for {coupling_type} ***')

    

    X_data, y_data, groups = build_x_y_data(train_csv, coupling_type, n_atoms)

    X_test, _, __ = build_x_y_data(test_csv, coupling_type, n_atoms)

    y_pred = np.zeros(X_test.shape[0], dtype='float32')

    

    cv_score = 0

    

    kfold = GroupKFold(n_splits=n_folds)



    for fold, (train_index, val_index) in enumerate(kfold.split(X_data, y_data, groups=groups)):



        X_train, X_val = X_data[train_index], X_data[val_index]

        y_train, y_val = y_data[train_index], y_data[val_index]

        



        model = LGBMRegressor(**LGB_PARAMS, n_estimators=15000, n_jobs = -1)

        model.fit(X_train, y_train, 

            eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',

            verbose=500, early_stopping_rounds=300)



        y_val_pred = model.predict(X_val)

        val_score = np.log(mean_absolute_error(y_val, y_val_pred))

        print(f'{coupling_type} Fold {fold}, logMAE: {val_score}')

        

        cv_score += val_score

        y_pred += model.predict(X_test)

        

        break

        

        

    submission.loc[test_csv['type'] == coupling_type, 'scalar_coupling_constant'] = y_pred

    return cv_score
model_params = {

    '1JHN': 7,

    '1JHC': 10,

    '2JHH': 9,

    '2JHN': 9,

    '2JHC': 9,

    '3JHH': 9,

    '3JHC': 10,

    '3JHN': 10

}

N_FOLDS = 7

submission = submission_csv.copy()



cv_scores = {}

for coupling_type in model_params.keys():

    cv_score = train_and_predict_for_one_coupling_type(

        coupling_type, submission, n_atoms=model_params[coupling_type], n_folds=N_FOLDS)

    cv_scores[coupling_type] = cv_score
pd.DataFrame({'type': list(cv_scores.keys()), 'cv_score': list(cv_scores.values())})
np.mean(list(cv_scores.values()))
submission[submission['scalar_coupling_constant'] == 0].shape
submission.to_csv(f'{SUBMISSIONS_PATH}/chem_submission_lgbv35.csv')