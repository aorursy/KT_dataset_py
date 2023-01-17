import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')

structures = pd.read_csv('../input/structures.csv')

potential_energy = pd.read_csv('../input/potential_energy.csv')

mulliken_charges = pd.read_csv('../input/mulliken_charges.csv')

scalar_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv')

magnetic_shielding_tensors = pd.read_csv('../input/magnetic_shielding_tensors.csv')

dipole_moments = pd.read_csv('../input/dipole_moments.csv')
train.head()
train.nunique()
train.isnull().sum()
train.info()
train.describe()
plt.hist(train['scalar_coupling_constant'], bins=50)

plt.title('count of scalar_coupling_constant')

plt.xlabel('scalar_coupling_constant')

plt.ylabel('count')

plt.show()
plt.figure(figsize=(12,4))

sns.stripplot(x=train.scalar_coupling_constant)

plt.show()
train.sort_values('scalar_coupling_constant', ascending=False).head(20)
plt.figure(figsize=(15, 5))

plt.title('violin plot of scalar_coupling_constant')

sns.violinplot(x='type', y='scalar_coupling_constant', data=train)

plt.show()
def count_plot(df, col):

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    ax.bar(df[col].unique(), df.groupby(col)[col].count())

    ax.set_xlabel(col)

    ax.set_ylabel('count')

    ax.set_title(f'count of {col}')

    fig.show()
count_plot(train, 'type')
count_plot(train, 'atom_index_0')
count_plot(train, 'atom_index_1')
test.head()
test.nunique()
test.isnull().sum()
test.info()
test.describe()
count_plot(test, 'type')
count_plot(test, 'atom_index_0')
count_plot(test, 'atom_index_1')
structures.head()
plt.figure(figsize=(25, 5))



plt.subplot(1,3,1)

plt.hist(structures['x'], bins=50)

plt.title('count of x')

plt.xlabel('x')

plt.ylabel('count')



plt.subplot(1,3,2)

plt.hist(structures['y'], bins=50)

plt.title('count of y')

plt.xlabel('y')

plt.ylabel('count')



plt.subplot(1,3,3)

plt.hist(structures['z'], bins=50)

plt.title('count of z')

plt.xlabel('z')

plt.ylabel('count')



plt.show()
potential_energy.head()
plt.hist(potential_energy['potential_energy'], bins=50)

plt.title('count of potential_energy')

plt.xlabel('potential_energy')

plt.ylabel('count')

plt.show()
mulliken_charges.head()
plt.hist(mulliken_charges['mulliken_charge'], bins=50)

plt.title('count of mulliken_charge')

plt.xlabel('mulliken_charge')

plt.ylabel('count')

plt.show()
plt.figure(figsize=(25, 5))

plt.title('violin plot of mulliken_charge')

sns.violinplot(x='atom_index', y='mulliken_charge', data=mulliken_charges)

plt.show()
scalar_coupling_contributions.head()
plt.figure(figsize=(18, 10))



plt.subplot(2,2,1)

plt.hist(scalar_coupling_contributions['fc'], bins=50)

plt.title('count of fc')

plt.xlabel('fc')

plt.ylabel('count')



plt.subplot(2,2,2)

plt.hist(scalar_coupling_contributions['sd'], bins=50)

plt.title('count of sd')

plt.xlabel('sd')

plt.ylabel('count')



plt.subplot(2,2,3)

plt.hist(scalar_coupling_contributions['pso'], bins=50)

plt.title('count of pso')

plt.xlabel('pso')

plt.ylabel('count')



plt.subplot(2,2,4)

plt.hist(scalar_coupling_contributions['dso'], bins=50)

plt.title('count of dso')

plt.xlabel('dso')

plt.ylabel('count')



plt.show()
plt.figure(figsize=(15, 5))

plt.title('violin plot of fc')

sns.violinplot(x='type', y='fc', data=scalar_coupling_contributions)

plt.show()
plt.figure(figsize=(15, 5))

plt.title('violin plot of sd')

sns.violinplot(x='type', y='sd', data=scalar_coupling_contributions)

plt.show()
plt.figure(figsize=(15, 5))

plt.title('violin plot of pso')

sns.violinplot(x='type', y='pso', data=scalar_coupling_contributions)

plt.show()
plt.figure(figsize=(15, 5))

plt.title('violin plot of dso')

sns.violinplot(x='type', y='dso', data=scalar_coupling_contributions)

plt.show()
magnetic_shielding_tensors.head()
dipole_moments.head()
matrix = pd.merge(train,

                  structures,

                  how = 'left',

                  left_on = ['molecule_name', 'atom_index_0'],

                  right_on = ['molecule_name', 'atom_index'])



matrix = matrix.drop('atom_index', axis=1)



matrix = matrix.rename(columns={'atom': 'atom_0',

                                'x': 'x_0',

                                'y': 'y_0',

                                'z': 'z_0'})
matrix = pd.merge(matrix,

                  structures,

                  how = 'left',

                  left_on = ['molecule_name', 'atom_index_1'],

                  right_on = ['molecule_name', 'atom_index'])



matrix = matrix.drop('atom_index', axis=1)



matrix = matrix.rename(columns={'atom': 'atom_1',

                                'x': 'x_1',

                                'y': 'y_1',

                                'z': 'z_1'})
matrix = pd.merge(matrix,

                  mulliken_charges,

                  how = 'left',

                  left_on = ['molecule_name', 'atom_index_0'],

                  right_on = ['molecule_name', 'atom_index'])



matrix = matrix.drop('atom_index', axis=1)



matrix = matrix.rename(columns={'mulliken_charge': 'mulliken_charge_0'})
matrix = pd.merge(matrix,

                  mulliken_charges,

                  how = 'left',

                  left_on = ['molecule_name', 'atom_index_1'],

                  right_on = ['molecule_name', 'atom_index'])



matrix = matrix.drop('atom_index', axis=1)



matrix = matrix.rename(columns={'mulliken_charge': 'mulliken_charge_1'})
matrix = pd.merge(matrix,

                  scalar_coupling_contributions,

                  how = 'left',

                  left_on = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],

                  right_on = ['molecule_name', 'atom_index_0', 'atom_index_1',  'type'])
matrix = pd.merge(matrix,

                  magnetic_shielding_tensors,

                  how = 'left',

                  left_on = ['molecule_name', 'atom_index_0'],

                  right_on = ['molecule_name', 'atom_index'])



matrix = matrix.drop('atom_index', axis=1)



matrix = matrix.rename(columns={'XX':'XX_0',

                                'YX':'YX_0',

                                'ZX':'ZX_0',

                                'XY':'XY_0',

                                'YY':'YY_0',

                                'ZY':'ZY_0',

                                'XZ':'XZ_0',

                                'YZ':'YZ_0',

                                'ZZ':'ZZ_0'})
matrix = pd.merge(matrix,

                  magnetic_shielding_tensors,

                  how = 'left',

                  left_on = ['molecule_name', 'atom_index_1'],

                  right_on = ['molecule_name', 'atom_index'])



matrix = matrix.drop('atom_index', axis=1)



matrix = matrix.rename(columns={'XX':'XX_1',

                                'YX':'YX_1',

                                'ZX':'ZX_1',

                                'XY':'XY_1',

                                'YY':'YY_1',

                                'ZY':'ZY_1',

                                'XZ':'XZ_1',

                                'YZ':'YZ_1',

                                'ZZ':'ZZ_1'})
matrix = pd.merge(matrix,

                  dipole_moments,

                  how = 'left',

                  left_on = ['molecule_name'],

                  right_on = ['molecule_name'])
matrix.head()
matrix.info()
def downsize_data(df):

    float64_cols = [c for c in df if df[c].dtype == "float64"]

    int64_int32_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]

    df[float64_cols] = df[float64_cols].astype(np.float32)

    df[int64_int32_cols] = df[int64_int32_cols].astype(np.int16)

    df.info()
downsize_data(matrix)
def heat_map(df):

    # calcurate correlation coefficient

    df_corr = df.corr()

    # show heat map

    plt.figure(figsize=(30, 30))

    hm = sns.heatmap(df_corr,

                    cbar=True,

                    annot=True,

                    square=True,

                    annot_kws={'size':15},

                    fmt='.2f')

    plt.tight_layout()

    plt.show()
sel_columns = [

    #'id',

    #'molecule_name', 

    #'atom_index_0',

    #'atom_index_1', 

    #'type',

     'scalar_coupling_constant',

    #'atom_0',

     'x_0',

     'y_0', 

     'z_0', 

    #'atom_1',

     'x_1',

     'y_1',

     'z_1', 

     'mulliken_charge_0', 

     'mulliken_charge_1', 

     'fc',

     'sd', 

     'pso', 

     'dso', 

     'XX_0',

     'YX_0',

     'ZX_0', 

     'XY_0',

     'YY_0', 

     'ZY_0',

     'XZ_0',

     'YZ_0',

     'ZZ_0',

     'XX_1', 

     'YX_1',

     'ZX_1', 

     'XY_1', 

     'YY_1',

     'ZY_1', 

     'XZ_1',

     'YZ_1', 

     'ZZ_1',

     'X',

     'Y', 

     'Z']
heat_map(matrix[sel_columns])