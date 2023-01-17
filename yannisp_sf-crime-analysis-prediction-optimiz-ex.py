from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
import lightgbm as lgb
import ast
import csv
import pickle
# Optional if you want to run it locally and inspect it in real time using Tensorboard
#from tensorboardX import SummaryWriter 

out_file = 'LGB.csv'
MAX_EVALS = 5 #This has been set to a small number for demonstration. Increase it!
N_FOLDS = 5
pbar = tqdm(total=MAX_EVALS, desc="Hyperopt")
# Loading the data
train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])

# Wrangling the dataset
train.drop_duplicates(inplace=True)
train.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

imp = SimpleImputer(strategy='mean')

for district in train['PdDistrict'].unique():
    train.loc[train['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(
        train.loc[train['PdDistrict'] == district, ['X', 'Y']])

# Feature Engineering
def feature_engineering(data):
    data['Date'] = pd.to_datetime(data['Dates'].dt.date)
    data['n_days'] = (
        data['Date'] - data['Date'].min()).apply(lambda x: x.days)
    data['Day'] = data['Dates'].dt.day
    data['DayOfWeek'] = data['Dates'].dt.weekday
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['Block'] = data['Address'].str.contains('block', case=False)
    
    data.drop(columns=['Dates','Date','Address'], inplace=True)
        
    return data

train = feature_engineering(train)
train.drop(columns=['Descript','Resolution'], inplace=True)

# Encoding Categorical Variables
le1 = LabelEncoder()
train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])

le2 = LabelEncoder()
y = le2.fit_transform(train.pop('Category'))

# Forming the dataset
train_set = lgb.Dataset(
    train, label=y, categorical_feature=['PdDistrict'], free_raw_data=False)
def param_flatten(d, params={}):
    """Function that accepts a dictionary with nested dictionaries and returns a flattened dictionary"""
    for key, value in d.items():
        if not isinstance(value, dict):
            params[key] = value
        else:
            param_flatten(value, params)
            
    return params
def objective(params, n_folds=N_FOLDS):
    """Objective function for LightGBM Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION
    ITERATION += 1
    
    # We need all the parameters in a flattened dictionary
    params = param_flatten(params)

    # Make sure parameters that need to be integers are integers
    for key, value in params.items():
        if key in ['num_leaves', 'min_data_in_leaf']:
            params[key] = int(value)
            
    print(params)

    # Perform n_folds cross validation.
    # If you download this notebook you can add callbacks=[logspy] to use Tensorboard
    try:
        cv_results = lgb.cv(
            params,
            train_set,
            num_boost_round=100,
            nfold=n_folds,
            early_stopping_rounds=10,
            metrics='multi_logloss')

        # Extract the best score
        loss = min(cv_results['multi_logloss-mean'])
        print('loss: ',loss)

        # Boosting rounds that returned the highest cv score
        epochs = np.argmin(cv_results['multi_logloss-mean']) + 1
        
        # Write to the csv file ('a' means append)
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([loss, params, ITERATION, epochs])

        pbar.update()

        # Dictionary with information for evaluation
        return {
            'loss': loss,
            'params': params,
            'iteration': ITERATION,
            'epochs': epochs,
            'status': STATUS_OK
        }
    except Exception as e:
        print('EXCEPTION\n')
        print(e)
        return{'status': 'fail'}
space = {
    'boosting':
    hp.choice('boosting', [
        {
            'boosting': 'gbdt',
            'max_delta_step': hp.quniform('gbdt_max_delta_step', 0, 2, 0.1),
            'min_data_in_leaf': hp.quniform('gbdt_min_data_in_leaf', 10, 30,
                                            1),
            'num_leaves': hp.quniform('gbdt_num_leaves', 20, 40, 1)
        },
        {
            'boosting': 'dart',
            'max_delta_step': hp.quniform('dart_max_delta_step', 0, 2, 0.1),
            'min_data_in_leaf': hp.quniform('dart_min_data_in_leaf', 10, 30,
                                            1),
            'num_leaves': hp.quniform('dart_num_leaves', 20, 40, 1),
        },
    ]),
    'objective':
    'multiclass',
    'num_class':
    39
}
def run_trials():
    """Function to run the trials and save the results after every iteration.
    This is usefull in case you need to interupt the execution and continue from where you left."""

    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 1  # initial max_trials. put something small to not have to wait

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("LGB.hyperopt", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(
            len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_trials,
        trials=trials)

    print("Best:", best)

    # save the trials object
    with open("LGB.hyperopt", "wb") as f:
        pickle.dump(trials, f)
#File to save first results

of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(
    ['loss', 'params', 'iteration', 'epochs'])
of_connection.close()
ITERATION = 0
while ITERATION <= MAX_EVALS:
    run_trials()
pbar.close()
trials = pickle.load(open("LGB.hyperopt", "rb"))
results = pd.DataFrame(trials.results)

bayes_params = pd.DataFrame(columns = list(results.loc[0, 'params'].keys()),
                            index = list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(params.values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']
bayes_params.sort_values('loss', inplace=True)

bayes_params.head()