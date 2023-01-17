# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import lightgbm as lgb

#from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.ensemble import RandomForestRegressor

print(os.listdir("../PredictingMolecularproperties/"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../PredictingMolecularproperties/train.csv')



df_test = pd.read_csv('../PredictingMolecularproperties/test.csv')

constant_data = pd.read_csv('../PredictingMolecularproperties/scalar_coupling_contributions.csv')

structures = pd.read_csv('../PredictingMolecularproperties/structures.csv')
ids = df_test["id"]
df_train.info()
df_train.type.unique()
from tqdm import tqdm_notebook as tqdm

atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor



fudge_factor = 0.05

atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}

print(atomic_radius)



electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}



#structures = pd.read_csv(structures, dtype={'atom_index':np.int8})



atoms = structures['atom'].values

atoms_en = [electronegativity[x] for x in tqdm(atoms)]

atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]



structures['EN'] = atoms_en

structures['rad'] = atoms_rad



display(structures.head())

i_atom = structures['atom_index'].values

p = structures[['x', 'y', 'z']].values

p_compare = p

m = structures['molecule_name'].values

m_compare = m

r = structures['rad'].values

r_compare = r



source_row = np.arange(len(structures))

max_atoms = 28



bonds = np.zeros((len(structures)+1, max_atoms+1), dtype=np.int8)

bond_dists = np.zeros((len(structures)+1, max_atoms+1), dtype=np.float32)



print('Calculating the bonds')



for i in tqdm(range(max_atoms-1)):

    p_compare = np.roll(p_compare, -1, axis=0)

    m_compare = np.roll(m_compare, -1, axis=0)

    r_compare = np.roll(r_compare, -1, axis=0)

    

    mask = np.where(m == m_compare, 1, 0) #Are we still comparing atoms in the same molecule?

    dists = np.linalg.norm(p - p_compare, axis=1) * mask

    r_bond = r + r_compare

    

    bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

    

    source_row = source_row

    target_row = source_row + i + 1 #Note: Will be out of bounds of bonds array for some values of i

    target_row = np.where(np.logical_or(target_row > len(structures), mask==0), len(structures), target_row) #If invalid target, write to dummy row

    

    source_atom = i_atom

    target_atom = i_atom + i + 1 #Note: Will be out of bounds of bonds array for some values of i

    target_atom = np.where(np.logical_or(target_atom > max_atoms, mask==0), max_atoms, target_atom) #If invalid target, write to dummy col

    

    bonds[(source_row, target_atom)] = bond

    bonds[(target_row, source_atom)] = bond

    bond_dists[(source_row, target_atom)] = dists

    bond_dists[(target_row, source_atom)] = dists



bonds = np.delete(bonds, axis=0, obj=-1) #Delete dummy row

bonds = np.delete(bonds, axis=1, obj=-1) #Delete dummy col

bond_dists = np.delete(bond_dists, axis=0, obj=-1) #Delete dummy row

bond_dists = np.delete(bond_dists, axis=1, obj=-1) #Delete dummy col



print('Counting and condensing bonds')



bonds_numeric = [[i for i,x in enumerate(row) if x] for row in tqdm(bonds)]

bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate(tqdm(bond_dists))]

bond_lengths_mean = [ np.mean(x) for x in bond_lengths]

n_bonds = [len(x) for x in bonds_numeric]



#bond_data = {'bond_' + str(i):col for i, col in enumerate(np.transpose(bonds))}

#bond_data.update({'bonds_numeric':bonds_numeric, 'n_bonds':n_bonds})



bond_data = {'n_bonds':n_bonds, 'bond_lengths_mean': bond_lengths_mean }

bond_df = pd.DataFrame(bond_data)

structures = structures.join(bond_df)

display(structures.head(20))
del bond_data

del bond_df

del bonds_numeric

del bond_lengths

del bond_lengths_mean

del i_atom

del p, p_compare

del m, m_compare

del r, r_compare

del source_row

del bonds

del dists

del atoms

del atoms_en

del atoms_rad

y = df_train['scalar_coupling_constant']
len_train = df_train.shape[0]

combined = pd.concat([df_train,df_test],sort=False)
structures['molecule_name_atom'] = structures['molecule_name'] + "_" + structures['atom_index'].apply(str)

combined['molecule_name_atom0'] = combined['molecule_name'] + "_" + combined['atom_index_0'].apply(str)

combined_f = pd.merge(combined, structures, how = 'left', left_on='molecule_name_atom0', right_on='molecule_name_atom')

#combined_f
combined_f.drop("molecule_name_atom0", axis=1, inplace=True)

combined_f.drop("molecule_name_y", axis=1, inplace=True)

combined_f.drop("atom_index", axis=1, inplace=True)

combined_f.drop("molecule_name_atom", axis=1, inplace=True)



combined_f.rename(columns = {"x":"x0",

                "EN":"EN0",

                "rad":"rad0",

                "n_bonds":"n_bonds0",

                "bond_lengths_mean":"bond_lengths_mean0",

                 "y":"y0",

                  "z":"z0",

                  "atom":"atom0",

                  "molecule_name_x":"molecule_name"

                 }, inplace=True)

#combined_f
combined_f['molecule_name_atom1'] = combined_f['molecule_name'] + "_" + combined_f['atom_index_1'].apply(str)

combined = pd.merge(combined_f, structures, how = 'left', left_on='molecule_name_atom1', right_on='molecule_name_atom')

#combined
combined.drop("molecule_name_atom1", axis=1, inplace=True)

combined.drop("molecule_name_y", axis=1, inplace=True)

combined.drop("atom_index", axis=1, inplace=True)

combined.drop("molecule_name_atom", axis=1, inplace=True)



combined.rename(columns = {"x":"x1",

                 "EN":"EN1",

                "rad":"rad1",

                "n_bonds":"n_bonds1",

                "bond_lengths_mean":"bond_lengths_mean1",

                 "y":"y1",

                  "z":"z1",

                  "atom":"atom1",

                  "molecule_name_x":"molecule_name"

                 }, inplace=True)

combined
combined.info()

del combined_f
combined["atom1"].unique()
pos_0 = combined[['x0', 'y0', 'z0']].values

pos_1 = combined[['x1', 'y1', 'z1']].values



combined['dist'] = np.linalg.norm(pos_0 - pos_1, axis=1)

combined['dist_x'] = (combined["x0"] - combined["x1"])**2

combined['dist_y'] = (combined["y0"] - combined["y1"])**2

combined['dist_z'] = (combined["z0"] - combined["z1"])**2





combined["atom_indexes"] = combined["atom_index_0"] + combined["atom_index_1"]

combined["distance^2"] = (combined["dist"])**2

combined["distance_sqrt"] = np.sqrt(combined["dist"])

combined["1/distance^3"] = 1 / (combined["dist"] + 1)**3

combined["distance^3"] = (combined["dist"])**3

combined['type_0'] = combined['type'].apply(lambda x: x[0])
combined['num_molecule_bonds'] = combined.groupby('molecule_name')['id'].transform('count')
combined['dist_mean'] = combined.groupby('molecule_name')['dist'].transform('mean')

combined['dist_min'] = combined.groupby('molecule_name')['dist'].transform('min')

combined['dist_max'] = combined.groupby('molecule_name')['dist'].transform('max')

combined['dist_std'] = combined.groupby('molecule_name')['dist'].transform('std')



combined['atom_0_num_bonds'] = combined.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')

combined['atom_1_num_bonds'] = combined.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')



combined["bf_dist_grouped_type"] = combined.groupby(['molecule_name', 'type'])["dist"].transform('mean')

combined["bf_dist_grouped_type"] = combined.groupby(['molecule_name', 'type'])["dist"].transform('max')

combined["bf_dist_grouped_type"] = combined.groupby(['molecule_name', 'type'])["dist"].transform('min')

combined["bf_dist_grouped_type"] = combined.groupby(['molecule_name', 'type'])["dist"].transform('std')



combined[f"bf_dist_grouped_n_bonds"] = combined.groupby(['molecule_name', 'n_bonds1'])["dist"].transform('mean')

combined[f"bf_dist_grouped_n_bonds"] = combined.groupby(['molecule_name', 'n_bonds1'])["dist"].transform('max')

combined[f"bf_dist_grouped_n_bonds"] = combined.groupby(['molecule_name', 'n_bonds1'])["dist"].transform('min')

combined[f"bf_dist_grouped_n_bonds"] = combined.groupby(['molecule_name', 'n_bonds1'])["dist"].transform('std')

        

combined[f"bf_dist_grouped_atom"] = combined.groupby(['molecule_name', 'atom1'])["dist"].transform('mean')

combined[f"bf_dist_grouped_atom1"] = combined.groupby(['molecule_name', 'atom1'])["dist"].transform('max')

combined[f"bf_dist_grouped_atom1"] = combined.groupby(['molecule_name', 'atom1'])["dist"].transform('min')

combined[f"bf_dist_grouped_atom1"] = combined.groupby(['molecule_name', 'atom1'])["dist"].transform('std')

        
combined['dist/maxdist'] = combined['dist'] / combined['dist_max']

combined['ENs/dist'] = (combined['EN0'] + combined['EN1']) / combined['dist']

combined['rads/dist'] = (combined['rad0'] + combined['rad1']) / combined['dist']

combined['ENS'] = combined['EN0'] + combined['EN1']

#combined['atom_num_bonds'] = combined['atom1'] + str(combined['n_bonds1'])
numerical = ["EN0", "rad0", "bond_lengths_mean0", "EN1", "rad1", "bond_lengths_mean1", "1/distance^3", "ENS"]

for col in numerical:

    for col2 in numerical:

        if col != col2:

            combined[f"bf_{col}_/_{col2}"] = combined[col] / combined[col2]

            combined[f"bf_{col}_*_{col2}"] = combined[col] * combined[col2]

            combined[f"bf_{col}_+_{col2}"] = combined[col] + combined[col2]

            combined[f"bf_{col}_-_{col2}"] = combined[col] - combined[col2]

            

        combined[f"bf_{col}_grouped_type"] = combined.groupby(['molecule_name', 'type'])[col].transform('mean')

        combined[f"bf_{col}_grouped_type"] = combined.groupby(['molecule_name', 'type'])[col].transform('max')

        combined[f"bf_{col}_grouped_type"] = combined.groupby(['molecule_name', 'type'])[col].transform('min')

        combined[f"bf_{col}_grouped_type"] = combined.groupby(['molecule_name', 'type'])[col].transform('std')



        combined[f"bf_{col}_grouped_n_bonds"] = combined.groupby(['molecule_name', 'n_bonds1'])[col].transform('mean')

        combined[f"bf_{col}_grouped_n_bonds"] = combined.groupby(['molecule_name', 'n_bonds1'])[col].transform('max')

        combined[f"bf_{col}_grouped_n_bonds"] = combined.groupby(['molecule_name', 'n_bonds1'])[col].transform('min')

        combined[f"bf_{col}_grouped_n_bonds"] = combined.groupby(['molecule_name', 'n_bonds1'])[col].transform('std')

        

        combined[f"bf_{col}_grouped_n_bonds"] = combined.groupby(['molecule_name', 'atom1'])[col].transform('mean')

        combined[f"bf_{col}_grouped_n_bonds"] = combined.groupby(['molecule_name', 'atom1'])[col].transform('max')

        combined[f"bf_{col}_grouped_n_bonds"] = combined.groupby(['molecule_name', 'atom1'])[col].transform('min')

        combined[f"bf_{col}_grouped_n_bonds"] = combined.groupby(['molecule_name', 'atom1'])[col].transform('std')

        



combined.info()

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
combined.drop("id", axis=1, inplace=True)

combined.drop("n_bonds0", axis=1, inplace=True)

combined.drop("atom0", axis=1, inplace=True)

combined.drop("molecule_name", axis=1, inplace=True)



combined["atom_indexes"] = combined["atom_indexes"].apply(str)

combined.drop("atom_index_0", axis=1, inplace=True)

combined.drop("atom_index_1", axis=1, inplace=True)

combined.drop("scalar_coupling_constant", axis=1, inplace=True)

#print(combined.head())

#combined = reduce_mem_usage(combined)

print(combined.head())

combined=pd.get_dummies(combined)


df_train=combined[:len_train]

df_test=combined[len_train:]

del len_train

del combined

del structures
x = df_train

#x = np.array(x)

#x = x.reshape((-1, 1))

y_fc = constant_data['fc']

#y_sd = constant_data['sd']

#y_pso = constant_data['pso']

#y_dso = constant_data['dso']

x_predict = df_test

#x_predict = np.array(x_predict)

#x_predict = x_predict.reshape((-1,1))

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8)



#df_train.info()

del constant_data
df_train.head()
df_train_1JHN = df_train[df_train["type_1JHN"] == 1]

y_fc_1JHN = y_fc[df_train["type_1JHN"] == 1]

y_1JHN = y[df_train["type_1JHN"] == 1]



df_train_1JHC_1 = df_train[(df_train["type_1JHC"] == 1) & (df_train["dist"] > 1.065)]

y_fc_1JHC_1 = y_fc[(df_train["type_1JHC"] == 1) & (df_train["dist"] > 1.065)]

y_1JHC_1 = y[(df_train["type_1JHC"] == 1) & (df_train["dist"] > 1.065)]



df_train_1JHC_2 = df_train[(df_train["type_1JHC"] == 1) & (df_train["dist"] <= 1.065)]

y_fc_1JHC_2 = y_fc[(df_train["type_1JHC"] == 1) & (df_train["dist"] <= 1.065)]

y_1JHC_2 = y[(df_train["type_1JHC"] == 1) & (df_train["dist"] <= 1.065)]



df_train_2JHC = df_train[df_train["type_2JHC"] == 1]

y_fc_2JHC = y_fc[df_train["type_2JHC"] == 1]

y_2JHC = y[df_train["type_2JHC"] == 1]



df_train_3JHH = df_train[df_train["type_3JHH"] == 1]

y_fc_3JHH = y_fc[df_train["type_3JHH"] == 1]

y_3JHH = y[df_train["type_3JHH"] == 1]



df_train_3JHC = df_train[df_train["type_3JHC"] == 1]

y_fc_3JHC = y_fc[df_train["type_3JHC"] == 1]

y_3JHC = y[df_train["type_3JHC"] == 1]



df_train_2JHH = df_train[df_train["type_2JHH"] == 1]

y_fc_2JHH = y_fc[df_train["type_2JHH"] == 1]

y_2JHH = y[df_train["type_2JHH"] == 1]



df_train_3JHN = df_train[df_train["type_3JHN"] == 1]

y_fc_3JHN = y_fc[df_train["type_3JHN"] == 1]

y_3JHN = y[df_train["type_3JHN"] == 1]



df_train_2JHN = df_train[df_train["type_2JHN"] == 1]

y_fc_2JHN = y_fc[df_train["type_2JHN"] == 1]

y_2JHN = y[df_train["type_2JHN"] == 1]

df_train_1JHN.info()
df_train_1JHN.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_train_1JHC_1.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_train_1JHC_2.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_train_2JHC.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_train_3JHH.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_train_3JHC.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_train_2JHH.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_train_3JHN.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_train_2JHN.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_test.insert(0, 'ID', range(1, 1 + len(df_test)))





df_test_1JHN = df_test[df_test["type_1JHN"] == 1]



df_test_1JHC_1 = df_test[(df_test["type_1JHC"] == 1) & (df_test["dist"] > 1.065)]



df_test_1JHC_2 = df_test[(df_test["type_1JHC"] == 1) & (df_test["dist"] <= 1.065)]



df_test_2JHC = df_test[df_test["type_2JHC"] == 1]



df_test_3JHH = df_test[df_test["type_3JHH"] == 1]



df_test_3JHC = df_test[df_test["type_3JHC"] == 1]



df_test_2JHH = df_test[df_test["type_2JHH"] == 1]



df_test_3JHN = df_test[df_test["type_3JHN"] == 1]



df_test_2JHN = df_test[df_test["type_2JHN"] == 1]

df_test_1JHN.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_test_1JHC_1.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_test_1JHC_2.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_test_2JHC.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_test_3JHH.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_test_3JHC.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_test_2JHH.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_test_3JHN.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)

df_test_2JHN.drop(["type_1JHC", "type_1JHN", "type_2JHC", "type_2JHH", "type_2JHN", "type_3JHC", "type_3JHH", "type_3JHN"], axis=1, inplace=True)
def train_type_wise(train_data, y_fc_type, y_type):

    d_train_fc = lgb.Dataset(train_data, label=y_fc_type)

    params = {}

    params['learning_rate'] = 0.2

    params['boosting_type'] = 'gbdt'

    params['objective'] = 'regression'

    params['metric'] = 'mae'

    params['num_leaves'] = 10

    params['min_data'] = 50

    params['max_depth'] = 10



    clf_fc = lgb.train(params, d_train_fc, 2000)

    

    del d_train_fc

    del params

    

    train_data["fc"] = clf_fc.predict(train_data)

    #del clf_fc

    

    

    d_train = lgb.Dataset(train_data, label=y_type)

    params = {'num_leaves': 128,

          'min_child_samples': 79,

          'objective': 'regression',

          'max_depth': 9,

          'learning_rate': 0.2,

          "boosting_type": "gbdt",

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1,

          'reg_lambda': 0.3,

         }

    cl = lgb.train(params, d_train, 2000)

    

    train_data["ridge"] = cl.predict(train_data)



    del d_train

    del params

    #del cl



    

    d_train = lgb.Dataset(train_data, label=y_type)

    params = {

                'boosting_type': 'gbdt',

                'objective': 'regression',

                'metric': 'mae',

                'learning_rate': 0.2,

                'num_leaves': 20, 

                'reg_alpha': 0.5, 

                'reg_lambda': 0.5, 

                'nthread': 4, 

                'device': 'cpu',

                'min_child_samples': 45

            }



    clf = lgb.train(params, d_train, 2000)

    return clf_fc, cl, clf

model_1JHN_fc, model_1JHN_1, model_1JHN_2 = train_type_wise(df_train_1JHN, y_fc_1JHN, y_1JHN)



model_1JHC_1_fc, model_1JHC_1_1, model_1JHC_1_2 = train_type_wise(df_train_1JHC_1, y_fc_1JHC_1, y_1JHC_1)



model_1JHC_2_fc, model_1JHC_2_1, model_1JHC_2_2 = train_type_wise(df_train_1JHC_2, y_fc_1JHC_2, y_1JHC_2)



model_2JHC_fc, model_2JHC_1, model_2JHC_2 = train_type_wise(df_train_2JHC, y_fc_2JHC, y_2JHC)



model_3JHH_fc, model_3JHH_1, model_3JHH_2 = train_type_wise(df_train_3JHH, y_fc_3JHH, y_3JHH)



model_3JHC_fc, model_3JHC_1, model_3JHC_2 = train_type_wise(df_train_3JHC, y_fc_3JHC, y_3JHC)



model_2JHH_fc, model_2JHH_1, model_2JHH_2 = train_type_wise(df_train_2JHH, y_fc_2JHH, y_2JHH)



model_3JHN_fc, model_3JHN_1, model_3JHN_2 = train_type_wise(df_train_3JHN, y_fc_3JHN, y_3JHN)



model_2JHN_fc, model_2JHN_1, model_2JHN_2 = train_type_wise(df_train_2JHN, y_fc_2JHN, y_2JHN)
def predict_type_wise(test_data, model_fc, model_1, model_2):

    x_pred = test_data.drop("ID", axis=1)

    test_data["fc"] = model_fc.predict(x_pred)

    x_pred = test_data.drop("ID", axis=1)

    test_data["ridge"] = model_1.predict(x_pred)

    x_pred = test_data.drop("ID", axis=1)

    test_data["predictions"] = model_2.predict(x_pred)

    return test_data
predictions_1JHN = predict_type_wise(df_test_1JHN, model_1JHN_fc, model_1JHN_1, model_1JHN_2)



predictions_1JHC_1 = predict_type_wise(df_test_1JHC_1, model_1JHC_1_fc, model_1JHC_1_1, model_1JHC_1_2)



predictions_1JHC_2 = predict_type_wise(df_test_1JHC_2, model_1JHC_2_fc, model_1JHC_2_1, model_1JHC_2_2)



predictions_2JHC = predict_type_wise(df_test_2JHC, model_2JHC_fc, model_2JHC_1, model_2JHC_2)



predictions_3JHH = predict_type_wise(df_test_3JHH, model_3JHH_fc, model_3JHH_1, model_3JHH_2)



predictions_3JHC = predict_type_wise(df_test_3JHC, model_3JHC_fc, model_3JHC_1, model_3JHC_2)



predictions_2JHH = predict_type_wise(df_test_2JHH, model_2JHH_fc, model_2JHH_1, model_2JHH_2)



predictions_3JHN = predict_type_wise(df_test_3JHN, model_3JHN_fc, model_3JHN_1, model_3JHN_2)



predictions_2JHN = predict_type_wise(df_test_2JHN, model_2JHN_fc, model_2JHN_1, model_2JHN_2)

predicted_data = pd.concat([predictions_1JHN, predictions_1JHC_1, predictions_1JHC_2, predictions_2JHC, predictions_3JHH, predictions_3JHC, predictions_2JHH, predictions_3JHN, predictions_2JHN], ignore_index=True)



predicted_data
final_data = pd.merge(df_test, predicted_data, how='left', left_on="ID", right_on="ID")
final_data.head()
#prediction_array = np.array(predictions_scalar)

#temp = prediction_array.flat

#predictions = list(temp)

#print(predictions)

predictions = final_data["predictions"]

my_sub = pd.DataFrame({'id':ids, 'scalar_coupling_constant':predictions})

my_sub.to_csv('submission.csv', index = False)