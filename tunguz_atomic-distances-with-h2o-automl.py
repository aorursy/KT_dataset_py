import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GroupKFold



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv', index_col='id')

test = pd.read_csv('../input/test.csv', index_col='id')
train.head()
train.shape
test.shape
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
%%time

# This block is SPPED UP



train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
molecules = train.pop('molecule_name')

test = test.drop('molecule_name', axis=1)
train.head()
train['fold'] = 0
n_splits = 3

gkf = GroupKFold(n_splits=n_splits) # we're going to split folds by molecules





for fold, (in_index, oof_index) in enumerate(gkf.split(train, groups=molecules)):

    train.loc[oof_index, 'fold'] = fold
import h2o

print(h2o.__version__)

from h2o.automl import H2OAutoML



h2o.init(max_mem_size='14G')
train = h2o.H2OFrame(train)
test = h2o.H2OFrame(test)
train['type'] = train['type'].asfactor()

train['atom_0'] = train['atom_0'].asfactor()

train['atom_1'] = train['atom_1'].asfactor()



test['type'] = test['type'].asfactor()

test['atom_0'] = test['atom_0'].asfactor()

test['atom_1'] = test['atom_1'].asfactor()
x = test.columns

y = 'scalar_coupling_constant'
aml = H2OAutoML(max_models=2, seed=47, max_runtime_secs=3600)

aml.train(x=x, y=y, training_frame=train)
# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
# The leader model is stored here

aml.leader
preds = aml.predict(test)

sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission['scalar_coupling_constant'] = preds.as_data_frame().values.flatten()

sample_submission.to_csv('h2o_submission_3.csv', index=False)