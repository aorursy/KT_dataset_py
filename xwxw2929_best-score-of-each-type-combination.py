import warnings

warnings.filterwarnings("ignore")

import numpy as np 

import pandas as pd

import os

print(os.listdir("../input"))
test = pd.read_csv('../input/champs-scalar-coupling/test.csv')

sub1 = pd.read_csv('../input/keras-neural-net-and-distance-features/submission.csv')

sub2 = pd.read_csv('../input/keras-nn-with-multi-output/submission.csv')

display(test.head(),sub1.head(),sub2.head())
sub = pd.DataFrame(columns = ['id','scalar_coupling_constant'])

mol_types1 = ['2JHH','2JHN','2JHC','3JHH', '3JHC', '3JHN']

mol_types2 = ['1JHC', '1JHN']
for mol_type in mol_types1:

    index = test[test['type']==mol_type].id

    temp= sub1[sub1['id'].isin(index)]

    sub = pd.concat([sub, temp])



for mol_type in mol_types2:

    index = test[test['type']==mol_type].id

    temp= sub2[sub2['id'].isin(index)]

    sub = pd.concat([sub, temp])

sub.shape
sub.sort_values(['id'], inplace = True)

sub.head()
sub.to_csv("/kaggle/working/submission.csv", index=False)