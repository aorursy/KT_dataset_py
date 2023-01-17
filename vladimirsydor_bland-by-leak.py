import gc

import os

import numpy as np

import pandas as pd

import random

import sys

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import plotly.offline as py



from pathlib import Path

from glob import glob

from tqdm import tqdm_notebook as tqdm 

from IPython.core.display import display, HTML

from plotly import tools, subplots

from sklearn.metrics import mean_squared_error





py.init_notebook_mode(connected=True)
def metric(y_true, y_pred):

    return mean_squared_error(np.log1p(y_true), np.log1p(y_pred))**(0.5)
USE_STACKING = False

USE_HYPEROPT = False

USE_GENETIC_ALG = True

EXCLUDE_LIST =[

    'bland-nn-on-pp-leaks-train-fe_version3',

    'bland-lgbt-on-leaks_version1',

    'bland-lgbt-on-leaks_version2',

    'ashrae-exploiting-leak-site-5',

    'ashrae-1-1-to-1-06-with-ucl',

    'ashrae-divide-and-conquer-fix0'

               ]



ADD_LIST = []


# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

# Modified to support timestamp type, categorical type

# Modified to add option to use float16 or not. feather format does not support float16.

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            # skip datetime type or categorical type

            continue

        col_type = df[col].dtype

        

        if col_type != object:

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

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
leak_df = reduce_mem_usage(pd.read_csv('/kaggle/input/leakaggregator/leaked_test_target.csv'))



not_nan_values_in_leak_df = ~leak_df['meter_reading'].isna()
not_nan_values_in_leak_df.value_counts(normalize=True)
leak_df.head()
datasets_to_take = [

    '/kaggle/input/ashrae-submissions/*.csv',

    '/kaggle/input/ashrae-submissions-1/*.csv',

    '/kaggle/input/ashrae-submissions-2/*.csv',

    '/kaggle/input/ashrae-submissions-3/*.csv',

    '/kaggle/input/ashrae-submissions-4/*.csv',

    '/kaggle/input/ashrae-submissions-5/*.csv',

    '/kaggle/input/ashrae-submissions-6/*.csv'

]



files_to_take = []

for dtt in datasets_to_take:

    files_to_take += glob(dtt)
all_sumbissions = pd.DataFrame()

all_sumbissions['row_id'] = leak_df['row_id']



if ADD_LIST:

    print('ADD MODE')

    for df_name in tqdm(files_to_take):

        if os.path.basename(df_name).split('.')[0] in ADD_LIST:

            print(df_name)

            all_sumbissions[os.path.basename(df_name).split('.')[0]] = reduce_mem_usage(pd.read_csv(df_name)).sort_values('row_id')['meter_reading']

            gc.collect()

else:

    print('EXCLUDE MODE')

    for df_name in tqdm(files_to_take):

        if os.path.basename(df_name).split('.')[0] in EXCLUDE_LIST:

            continue

        print(df_name)

        all_sumbissions[os.path.basename(df_name).split('.')[0]] = reduce_mem_usage(pd.read_csv(df_name)).sort_values('row_id')['meter_reading']

        gc.collect()
gc.collect()
all_sumbissions.columns
for col in all_sumbissions.columns[all_sumbissions.min() < 0]:

    all_sumbissions.loc[all_sumbissions[col] < 0, col] = 0
gc.collect()
for col in all_sumbissions.columns[1:]:

    print(col)

    leak_score = metric(leak_df.loc[not_nan_values_in_leak_df, 'meter_reading'],

                        all_sumbissions.loc[not_nan_values_in_leak_df, col])

    print ('score1=', leak_score)

    gc.collect()
leak_score = metric(leak_df.loc[not_nan_values_in_leak_df, 'meter_reading'], all_sumbissions.loc[not_nan_values_in_leak_df].iloc[:,1:].mean(axis=1))

print ('mean score=', leak_score) 
leak_score = metric(leak_df.loc[not_nan_values_in_leak_df, 'meter_reading'], all_sumbissions.loc[not_nan_values_in_leak_df].iloc[:,1:].median(axis=1))

print ('mean score=', leak_score) # 0.963450549517071
from itertools import permutations



class GenerativeAlgorithm(object):

    def __init__(self, 

                 x_shape,

                 amout_ind_to_mutate,

                 population_size, 

                 target_function, 

                 num_generations, 

                 populations_to_take,

                 available_indices,

                 mode='min'):

        self.amout_ind_to_mutate = amout_ind_to_mutate

        self.x_shape = x_shape

        self.population_size = population_size

        self.target_function = target_function

        self.num_generations = num_generations

        self.populations_to_take = populations_to_take

        self.available_indices = available_indices

        self.mode = mode

        

        self.gen_initial_generation()

        

        if self.mode == 'min':

            self.best_score = 1000000

        if self.mode == 'max':

            self.best_score = -1000000

            

    def gen_initial_generation(self):

        self.initial_generation = []

        for i, el in enumerate(permutations(self.available_indices, self.x_shape)):

            if i == self.population_size:

                break

            self.initial_generation.append(list(el))



        self.initial_generation = np.array(self.initial_generation)

        

    def select_mating_pool(self, population, fitness):

        

        parents = np.empty((self.populations_to_take, population.shape[1]))



        for parent_num in range(self.populations_to_take):

            

            if self.mode == 'min':

                fitness_idx = np.argmin(fitness)

            elif self.mode == 'max':

                fitness_idx = np.argmax(fitness)

            else:

                raise ValueError('Not implemented!')

                

            parents[parent_num, :] = population[fitness_idx, :]

            

                        

            if self.mode == 'min':

                fitness[fitness_idx] = 1000000

            elif self.mode == 'max':

                fitness[fitness_idx] = - 1000000



        return parents

    

    def crossover(self, parents, offspring_size):

        offspring = np.empty(offspring_size)

        crossover_point = np.uint8(offspring_size[1]/2)



        for k in range(offspring_size[0]):

            parent1_idx = np.random.randint(0, parents.shape[0])

            parent2_idx = np.random.randint(0, parents.shape[0])



            offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]

            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

            

        return offspring



    def mutation(self, offspring_crossover):



        for idx in range(offspring_crossover.shape[0]):

            

            

            for i in range(self.amout_ind_to_mutate):

                cur_available_idices = list(set(self.available_indices) - set(offspring_crossover[idx,:]))

                

                value = np.random.choice(cur_available_idices)

                idx_1 = np.random.randint(0,offspring_crossover.shape[1])

            

                offspring_crossover[idx, idx_1] = value

                                

        return offspring_crossover



    def apply_function_on_array(self, ar):

        return np.array([self.target_function(ar[i,:]) for i in range(ar.shape[0])])



    def fit(self):

        cur_population = self.initial_generation

        history = {'best_points':[], 'best_fits':[]}

        generation = 0

        

        while generation < self.num_generations:



            fitness = self.apply_function_on_array(cur_population)

                        

            if self.mode == 'min':

                best_idx = np.argmin(fitness)

            elif self.mode == 'max':

                best_idx = np.argmax(fitness)

                

            history['best_points'].append(cur_population[best_idx,:])

            history['best_fits'].append(fitness[best_idx])

            self.best_score = fitness[best_idx]

            

            print('Itteration {} best score: {}'.format(generation, self.best_score))

            

            parents = self.select_mating_pool(cur_population, fitness)

            

            offspring_crossover = self.crossover(parents,

                                                 offspring_size=(self.population_size-parents.shape[0], cur_population.shape[1]))

            

            offspring_mutation = self.mutation(offspring_crossover)

            

            cur_population[:parents.shape[0], :] = parents

            cur_population[parents.shape[0]:, :] = offspring_mutation

                        

            generation+=1 

            

        fitness = self.apply_function_on_array(cur_population)

        

        if self.mode == 'min':

            best_idx = np.argmin(fitness)

        elif self.mode == 'max':

            best_idx == np.argmax(fitness)

            

        history['best_points'].append(cur_population[best_idx,:])

        history['best_fits'].append(fitness[best_idx])

            

        return history
if USE_GENETIC_ALG:

    def tgt_f(col_idx):

        sc = metric(leak_df.loc[not_nan_values_in_leak_df, 'meter_reading'], 

                                   all_sumbissions.loc[not_nan_values_in_leak_df].iloc[:,col_idx].median(axis=1))

        return sc





    a_ind = np.array(range(1, all_sumbissions.shape[1]-1))

    

    alg = GenerativeAlgorithm(

    amout_ind_to_mutate=6,

    population_size=20,

    x_shape=11,

    target_function=tgt_f, 

    num_generations=2, 

    populations_to_take=6,

    available_indices=a_ind

    )

    

    h = alg.fit()
if USE_GENETIC_ALG:

    best_cols = h['best_points'][-1]

    

    m = metric(leak_df.loc[not_nan_values_in_leak_df, 'meter_reading'], all_sumbissions.loc[not_nan_values_in_leak_df].iloc[:,best_cols].median(axis=1))

    print('best_comb: {}\nbest_metric: {}'.format(all_sumbissions.columns[best_cols], m))
if USE_HYPEROPT:

    import hyperopt as hp

    

    score_comb = []

    

    def objective(x):

        cols_idx = [el[1] for el in x]

        

        if len(set(cols_idx)) != len(cols_idx):

            return 100

        

        m = metric(leak_df.loc[not_nan_values_in_leak_df, 'meter_reading'], all_sumbissions[not_nan_values_in_leak_df].iloc[:,cols_idx].median(axis=1))

        gc.collect()

        score_comb.append([cols_idx,m])

        return m

    

    space = [(str(i),1 + hp.hp.randint(str(i), all_sumbissions.shape[1]-1)) for i in range(10)]



    best = hp.fmin(objective, space, algo=hp.tpe.suggest, max_evals=1000)
if USE_HYPEROPT:

    best_cols = pd.DataFrame(score_comb).sort_values(1).iloc[0,0]

    

    m = metric(leak_df.loc[not_nan_values_in_leak_df, 'meter_reading'], all_sumbissions.loc[not_nan_values_in_leak_df].iloc[:,best_cols].median(axis=1))

    print('best_comb: {}\nbest_metric: {}'.format(all_sumbissions.columns[best_cols], m))
if USE_STACKING:

    for col in all_sumbissions.columns[1:]:

        all_sumbissions[col] = np.log1p(all_sumbissions[col])



    leak_df['meter_reading'] = np.log1p(leak_df['meter_reading'])
gc.collect()
if USE_STACKING:

    all_sumbissions['target'] = leak_df['meter_reading']



    del leak_df

    gc.collect()
if USE_STACKING:

    train = all_sumbissions[not_nan_values_in_leak_df]



    test = all_sumbissions[~not_nan_values_in_leak_df]



    del all_sumbissions

    gc.collect()
if USE_STACKING:

    train = train.set_index('row_id')

    test = test.set_index('row_id')
gc.collect()
class NNModel(object):

    def __init__(self, model, target_variable='target'):

        self.model = model

        

        self.es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=EARLY_STOPPING, verbose=False, mode='auto', restore_best_weights=True)

        self.rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, mode='auto', verbose=False)

        

        self.target_variable = target_variable

        

        self.scaling_stats = {}

        self.predictors = ['bland-lgbt-folds_version_1', 'bland-lgbt-folds_version_2',

       'pp-lgbt-refactored_version_7',

       'bland-lgbt-on-pp-leaks-train-fe_version1',

       'bland-nn-on-pp-leaks-train-fe_version1',

       'Bland LGBT on PP  Leaks Train  FE_version4',

       'bland-nn-on-pp-leaks-train-fe_version2',

       'Bland LGBT on PP  Leaks Train  FE_version2']

        

    def train_preprocessing(self, data):



        y = data[self.target_variable]

        

        data = data[self.predictors]

        print(data.shape)

        

        for col in self.predictors:

            self.scaling_stats[col] = {'mean':data[col].mean(), 'std':data[col].std()}

            data[col] = (data[col] - self.scaling_stats[col]['mean']) / self.scaling_stats[col]['std']

            

        print(data.isna().sum().sum())

        print('Scaling completed!')

        

        gc.collect()

        

        return data.values, y.values

    

    def test_preprocessing(self, data, is_val=False):

        

        if is_val:

            y = data[self.target_variable]

            

        data = data[self.predictors]

        gc.collect()

        

        for col in self.predictors:

            data[col] = (data[col] - self.scaling_stats[col]['mean']) / self.scaling_stats[col]['std']

        

        gc.collect()

        if is_val:

            return data.values, y.values

        else:

            return data.values

        

    def fit(self, data, data_val):

        

        data, y_train = self.train_preprocessing(data)

        data_val, y_val = self.test_preprocessing(data_val, is_val=True)

        

        self.model.fit(

            data, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(data_val, y_val), verbose=True, callbacks=[self.es, self.rlr]

        )

            

    def predict(self, data):

        

        data = self.test_preprocessing(data, is_val=False)

        

        return self.model.predict(data, batch_size=BATCH_SIZE, verbose=True).flatten()
from sklearn.model_selection import KFold



def time_val(data, model_create_func, metric_to_use=mean_squared_error, target_var_name='target', test_to_predict=None):

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    

    print('Starting Validation')

    results = []

    data['pred'] = 0

    if test_to_predict is not None:

        test_prediction = []

        

    for train_idx, test_idx in kf.split(data):

        print('New Itter')

        model = model_create_func()

        model.fit(data.iloc[train_idx].reset_index(drop=True), 

                  data.iloc[test_idx].reset_index(drop=True))

        

        gc.collect()

        

        data['pred'].iloc[test_idx] = model.predict(data.iloc[test_idx].reset_index(drop=True))

        

        gc.collect()

        itter_metric = metric_to_use(data.iloc[test_idx][target_var_name], data['pred'].iloc[test_idx])

        print('Itter metric: '+str(itter_metric))

        results.append(itter_metric)

        

        gc.collect()

        

        if test_to_predict is not None:

            test_prediction.append(model.predict(test_to_predict))

        

        gc.collect()

     

    if test_to_predict is not None:

        return results, sum(test_prediction)/NUM_FOLDS

    else:

        return results

from tqdm import tqdm



import keras.backend as K



from keras.layers import Input, Dense, Dropout, Embedding, Concatenate, Lambda

from keras.models import Model

from keras.optimizers import Adam, Nadam

from keras import callbacks



def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true)))



def create_model(inp_dim):

    inps = Input(shape=(inp_dim,))

    

    x = Dense(1)(inps)

    model = Model(inputs=inps, outputs=x)

    model.compile(

        optimizer=Nadam(lr=1e-3),

        loss=root_mean_squared_error

    )

    return model
gc.collect()
if USE_STACKING:

    NUM_FOLDS = 5

    N_EPOCHS = 3

    BATCH_SIZE = 256

    EARLY_STOPPING = 2



    my_model_create_f = lambda : NNModel(model = create_model(8))
if USE_STACKING:

    rf_res, test['target'] = time_val(train, my_model_create_f, test_to_predict=test)
if USE_STACKING:

    print('Result: {} +/- {}'.format(round(np.mean(rf_res),5), round(np.std(rf_res),5)))

    print(mean_squared_error(train['pred'], train['target']))
if USE_STACKING:

    train = train.rename(columns={'pred':'meter_reading'})

    test = test.rename(columns={'target':'meter_reading'})



    gc.collect()



    sub = pd.concat([train['meter_reading'], test['meter_reading']], axis=0)

    sub = sub.reset_index()



    sub['meter_reading'] = np.expm1(sub['meter_reading'])

    sub.loc[sub['meter_reading'] < 0, 'meter_reading'] = 0



    sub = sub.sort_values('row_id')
if USE_GENETIC_ALG or USE_HYPEROPT:

    all_sumbissions['meter_reading'] = all_sumbissions.iloc[:,best_cols].median(axis=1)

    all_sumbissions = all_sumbissions[['row_id','meter_reading']]

    gc.collect()

    all_sumbissions[['row_id','meter_reading']].to_csv('submission_blend.csv', index=False)

elif USE_STACKING:

    sub.to_csv('submission_blend.csv', index=False)

else:

    raise ValueError('Not ready')