import numpy as np

import pandas as pd

import os

import time

import datetime

import gc



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GroupKFold

from sklearn.metrics import mean_absolute_error



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook as tqdm



from catboost import CatBoostRegressor, Pool



import warnings

warnings.filterwarnings("ignore")
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



def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()



def encode_categoric(df):

    lbl = LabelEncoder()

    cat_cols=[]

    try:

        cat_cols = df.describe(include=['O']).columns.tolist()

        for cat in cat_cols:

            df[cat] = lbl.fit_transform(list(df[cat].values))

    except Exception as e:

        print('error: ', str(e) )



    return df
train = pd.read_csv('../input/train.csv')

structures = pd.read_csv('../input/structures.csv')



print('Train dataset shape is -> rows: {} cols:{}'.format(train.shape[0],train.shape[1]))

print('Structures dataset shape is  -> rows: {} cols:{}'.format(structures.shape[0],structures.shape[1]))
unique_molecules = train['molecule_name'].unique()



print("Few examples of molecule's names: ", '  '.join(unique_molecules[:3]), end='\n\n')

print('Amount of unique molecules in train: ', len(unique_molecules))
molecules_fraction = 0.1

molecules_amount = int(molecules_fraction * len(unique_molecules))



np.random.shuffle(unique_molecules)

train_molecules = unique_molecules[:molecules_amount]



train = train[train['molecule_name'].isin(train_molecules)]



print(f'Amount of molecules in the subset of train: {molecules_amount}, samples: {train.shape[0]}')
def atomic_radius_electonegativety(structures):

    atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor

    fudge_factor = 0.05

    atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}



    electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}



    atoms = structures['atom'].values

    atoms_en = [electronegativity[x] for x in atoms]

    atoms_rad = [atomic_radius[x] for x in atoms]



    structures['EN'] = atoms_en

    structures['rad'] = atoms_rad

    

    return structures





def create_bonds(structures):

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



#     print('Calculating bonds')



    for i in range(max_atoms-1):

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



#     print('Counting and condensing bonds')



    bonds_numeric = [[i for i,x in enumerate(row) if x] for row in bonds]

    bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate(bond_dists)]

    bond_lengths_mean = [ np.mean(x) for x in bond_lengths]

    n_bonds = [len(x) for x in bonds_numeric]





    bond_data = {'n_bonds':n_bonds, 'bond_lengths_mean': bond_lengths_mean }

    bond_df = pd.DataFrame(bond_data)

    structures = structures.join(bond_df)

    

    return structures



def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}',

                            'EN': f'EN_{atom_idx}',

                            'rad': f'rad_{atom_idx}',

                            'n_bonds': f'n_bonds_{atom_idx}',

                            'bond_lengths_mean': f'bond_lengths_mean_{atom_idx}',

                           })

    return df
structures = atomic_radius_electonegativety(structures)

structures = create_bonds(structures)



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



train.head()
def distances(df):

    df_p_0 = df[['x_0', 'y_0', 'z_0']].values

    df_p_1 = df[['x_1', 'y_1', 'z_1']].values

    

    df['dist'] = np.linalg.norm(df_p_0 - df_p_1, axis=1)

    df['dist_x'] = (df['x_0'] - df['x_1']) ** 2

    df['dist_y'] = (df['y_0'] - df['y_1']) ** 2

    df['dist_z'] = (df['z_0'] - df['z_1']) ** 2

    

    df['type_0'] = df['type'].apply(lambda x: x[0])

    

    return df



def map_atom_info(df_1,df_2, atom_idx):

    df = pd.merge(df_1, df_2, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    df = df.drop('atom_index', axis=1)



    return df



def create_closest(df):

    df_temp=df.loc[:,["molecule_name","atom_index_0","atom_index_1","dist","x_0","y_0","z_0","x_1","y_1","z_1"]].copy()

    df_temp_=df_temp.copy()

    df_temp_= df_temp_.rename(columns={'atom_index_0': 'atom_index_1',

                                       'atom_index_1': 'atom_index_0',

                                       'x_0': 'x_1',

                                       'y_0': 'y_1',

                                       'z_0': 'z_1',

                                       'x_1': 'x_0',

                                       'y_1': 'y_0',

                                       'z_1': 'z_0'})

    df_temp=pd.concat(objs=[df_temp,df_temp_],axis=0)



    df_temp["min_distance"]=df_temp.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')

    df_temp= df_temp[df_temp["min_distance"]==df_temp["dist"]]



    df_temp=df_temp.drop(['x_0','y_0','z_0','min_distance', 'dist'], axis=1)

    df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',

                                     'atom_index_1': 'atom_index_closest',

                                     'distance': 'distance_closest',

                                     'x_1': 'x_closest',

                                     'y_1': 'y_closest',

                                     'z_1': 'z_closest'})



    for atom_idx in [0,1]:

        df = map_atom_info(df,df_temp, atom_idx)

        df = df.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',

                                        'distance_closest': f'distance_closest_{atom_idx}',

                                        'x_closest': f'x_closest_{atom_idx}',

                                        'y_closest': f'y_closest_{atom_idx}',

                                        'z_closest': f'z_closest_{atom_idx}'})

    return df



def add_cos_features(df):

    df["distance_0"]=((df['x_0']-df['x_closest_0'])**2+(df['y_0']-df['y_closest_0'])**2+(df['z_0']-df['z_closest_0'])**2)**(1/2)

    df["distance_1"]=((df['x_1']-df['x_closest_1'])**2+(df['y_1']-df['y_closest_1'])**2+(df['z_1']-df['z_closest_1'])**2)**(1/2)

    df["vec_0_x"]=(df['x_0']-df['x_closest_0'])/df["distance_0"]

    df["vec_0_y"]=(df['y_0']-df['y_closest_0'])/df["distance_0"]

    df["vec_0_z"]=(df['z_0']-df['z_closest_0'])/df["distance_0"]

    df["vec_1_x"]=(df['x_1']-df['x_closest_1'])/df["distance_1"]

    df["vec_1_y"]=(df['y_1']-df['y_closest_1'])/df["distance_1"]

    df["vec_1_z"]=(df['z_1']-df['z_closest_1'])/df["distance_1"]

    df["vec_x"]=(df['x_1']-df['x_0'])/df["dist"]

    df["vec_y"]=(df['y_1']-df['y_0'])/df["dist"]

    df["vec_z"]=(df['z_1']-df['z_0'])/df["dist"]

    df["cos_0_1"]=df["vec_0_x"]*df["vec_1_x"]+df["vec_0_y"]*df["vec_1_y"]+df["vec_0_z"]*df["vec_1_z"]

    df["cos_0"]=df["vec_0_x"]*df["vec_x"]+df["vec_0_y"]*df["vec_y"]+df["vec_0_z"]*df["vec_z"]

    df["cos_1"]=df["vec_1_x"]*df["vec_x"]+df["vec_1_y"]*df["vec_y"]+df["vec_1_z"]*df["vec_z"]

    df=df.drop(['vec_0_x','vec_0_y','vec_0_z','vec_1_x','vec_1_y','vec_1_z','vec_x','vec_y','vec_z'], axis=1)

    return df



def create_features(df):

    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')

    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')

    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')

    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')

    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')

    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')

    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')

    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']

    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')

    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')

    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')

    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')

    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']

    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']

    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')

    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']

    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']

    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')

    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']

    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']

    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')

    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']

    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']

    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')

    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']

    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']

    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')

    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']

    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']

    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')

    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']

    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']

    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')

    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']

    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']

    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')

    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')

    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']

    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']

    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')

    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']

    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')

    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']

    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')

    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']

    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']

    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')

    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')

    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')

    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']

    return df
start_time = time.time()



train = distances(train)



print('Create closest features')



train = create_closest(train)



print('Create cos features')



train = add_cos_features(train)



print('Create groupby features', end='\n\n')



train = create_features(train)



train = reduce_mem_usage(train, verbose=False)



print('Train dataset shape is -> rows: {} cols:{}'.format(train.shape[0],train.shape[1]))

print('Structures dataset shape is  -> rows: {} cols:{}'.format(structures.shape[0],structures.shape[1]), end='\n\n')

print(f'Exe time: {(time.time() - start_time)/60:.2} min')
molecules_id = train['molecule_name']

X = train.drop(['id', 'scalar_coupling_constant', 'molecule_name'], axis=1)

y = train['scalar_coupling_constant']



X = encode_categoric(X)
print('X size', X.shape)



del train

gc.collect()

X.head()
kf = GroupKFold(4)

for tr_idx, val_idx in kf.split(X, groups=molecules_id):

    tr_X = X.iloc[tr_idx]; val_X = X.iloc[val_idx]

    tr_y = y.iloc[tr_idx]; val_y = y.iloc[val_idx]

    

    break
def permutation_importance(model, X_val, y_val, metric, threshold=0.005,

                           minimize=True, verbose=True):

    results = {}

    

    y_pred = model.predict(X_val)

    

    results['base_score'] = metric(y_val, y_pred)

    if verbose:

        print(f'Base score {results["base_score"]:.5}')



    

    for col in tqdm(X_val.columns):

        freezed_col = X_val[col].copy()



        X_val[col] = np.random.permutation(X_val[col])

        preds = model.predict(X_val)

        results[col] = metric(y_val, preds)



        X_val[col] = freezed_col

        

        if verbose:

            print(f'column: {col} - {results[col]:.5}')

    

    if minimize:

        bad_features = [k for k in results if results[k] < results['base_score'] + threshold]

    else:

        bad_features = [k for k in results if results[k] > results['base_score'] + threshold]

    bad_features.remove('base_score')

    

    return results, bad_features
def catboost_fit(model, X_train, y_train, X_val, y_val):

    train_pool = Pool(X_train, y_train)

    val_pool = Pool(X_val, y_val)

    model.fit(train_pool, eval_set=val_pool)

    

    return model



model = CatBoostRegressor(iterations=20000, 

                          max_depth=9,

                          objective='MAE',

                          task_type='GPU',

                          verbose=False)

model = catboost_fit(model, tr_X, tr_y, val_X, val_y)
from functools import partial

metric = partial(group_mean_log_mae, types=val_X['type'])
results, bad_features = permutation_importance(model=model,

                                               X_val=val_X,

                                               y_val=val_y,

                                               metric=metric,

                                               verbose=False)
results
bad_features
tr_X_reduced = tr_X.drop(bad_features, axis=1).copy()

val_X_reduced = val_X.drop(bad_features, axis=1).copy()
model_reduced = CatBoostRegressor(iterations=20000, 

                          max_depth=9,

                          objective='MAE',

                          task_type='GPU',

                          verbose=False)

model_reduced = catboost_fit(model, tr_X_reduced, tr_y, val_X_reduced, val_y)



y_pred = model_reduced.predict(val_X_reduced)

new_score = metric(val_y, y_pred)



print(f'Original score: {results["base_score"]:.3}, amount of features: {len(results)-1}')

print(f'Score after removing bad_features: {new_score:.3}, amount of features: {tr_X_reduced.shape[1]}')
from eli5.permutation_importance import get_score_importances



def score(X, y):

    y_pred = model.predict(X)

    return metric(y, y_pred)



base_score, score_decreases = get_score_importances(score, np.array(val_X), val_y, n_iter=1)



threshold = 0.001

bad_features = val_X.columns[score_decreases[0] > -threshold]
tr_X_reduced = tr_X.drop(bad_features, axis=1).copy()

val_X_reduced = val_X.drop(bad_features, axis=1).copy()



model_reduced = CatBoostRegressor(iterations=20000, 

                          max_depth=9,

                          objective='MAE',

                          task_type='GPU',

                          verbose=False)

model_reduced = catboost_fit(model_reduced, tr_X_reduced, tr_y, val_X_reduced, val_y)



y_pred = model_reduced.predict(val_X_reduced)

new_score = metric(val_y, y_pred)



print(f'Original score: {base_score:.3}, amount of features: {len(results)-1}')

print(f'Score after removing bad_features: {new_score:.3}, amount of features: {val_X_reduced.shape[1]}')