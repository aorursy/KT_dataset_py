# !pip install gplearn

import gplearn

print('ok')
import os

import shutil

import copy

import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_error

from gplearn.functions import make_function

from gplearn.genetic import SymbolicRegressor

from sklearn.model_selection import KFold

print('ok')
print(os.listdir('../..'))

print(os.listdir('../input'))

print(os.listdir('../input/gppyfiles'))
DATA_DIR = '../input/gplearn-data'

submission = pd.read_csv(os.path.join('../input/LANL-Earthquake-Prediction', 'sample_submission.csv'), index_col='seg_id')

scaled_train_X = pd.read_csv(os.path.join(DATA_DIR, 'scaled_train_X_AF0.csv'))

scaled_test_X = pd.read_csv(os.path.join(DATA_DIR, 'scaled_test_X_AF0.csv'))

train_y = pd.read_csv(os.path.join(DATA_DIR, 'train_y_AF0.csv'))

predictions = np.zeros(len(scaled_test_X))

print('ok')
from scipy.stats import pearsonr

y = train_y['time_to_failure'].values

pcol = []

pcor = []

pval = []

for col in scaled_train_X.columns:

    pcol.append(col)

    pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))

    pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))



df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))

df.sort_values(by=['cor', 'pval'], inplace=True)

df.dropna(inplace=True)

df = df.loc[df['pval'] <= 0.05]



drop_cols = []



for col in scaled_train_X.columns:

    if col not in df['col'].tolist():

        drop_cols.append(col)



scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)

scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

print(scaled_train_X.shape, scaled_test_X.shape)
# in an IDE this works as a lambda - not so in colab - have not tried a lambda on kaggle

from gplearn.functions import make_function



def th(x):

    return np.tanh(x)



gptanh = make_function(th, 'tanh', 1)

print('ok')
# import os

# import sys

# sys.path.append('../input/gppyfiles')

# from gppyfiles.gplearn_tanh import gptanh

function_set = ['add', 'sub', 'mul', 'div', 'inv', 'abs', 'neg', 'max', 'min', gptanh]  # 'sqrt', 'log', 

print('ok')
predictions = np.zeros(len(scaled_test_X))



n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)



fold_importance_df = pd.DataFrame()

fold_importance_df["Feature"] = scaled_train_X.columns

print('ok')
sample_wts = np.sqrt(np.array([x - 10.0 if x > 10.0 else 0 for x in y]) + 1.0)

print(y[0:16])

print(sample_wts[-8:])

print('ok')
# still needs cleanup

# GENS = 500

# MAE_THRESH = 2.5

# MAX_NO_IMPROVE = 50

# np.random.seed(666)

# maes = []

# gens = []



# for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):

#   print('working fold %d' % fold_)

#   X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]

#   y_tr, y_val = train_y['time_to_failure'].values[trn_idx].ravel(), train_y['time_to_failure'].values[val_idx].ravel()

#   sample_wts_tr = sample_wts[trn_idx]

#   np.random.seed(5591 + fold_)

#   best = 1e10

#   count = 1

#   imp_count = 0

#   best_mdl = None

#   best_iter = 0

  

#   gp = SymbolicRegressor(population_size=2000,

#                        generations=count,

#                        tournament_size=50,  # consider 20, was 50

#                        parsimony_coefficient=0.0001,  # oops: 0.0001?

#                        const_range=(-16, 16),  # consider +/-20, was 100

#                        function_set=function_set,

#                        # stopping_criteria=1.0,

#                        # p_hoist_mutation=0.05,

#                        # max_samples=.875,  # was in

#                        # p_crossover=0.7,

#                        # p_subtree_mutation=0.1,

#                        # p_point_mutation=0.1,

#                        init_depth=(6, 16),

#                        warm_start=True,

#                        metric='mean absolute error', verbose=1, random_state=42, n_jobs=-1, low_memory=True)



#   for run in range(GENS):

#       mdl = gp.fit(X_tr, y_tr, sample_weight=sample_wts_tr)

#       pred = gp.predict(X_val)

#       mae = np.sqrt(mean_absolute_error(y_val, pred))



#       if mae < best and imp_count < MAX_NO_IMPROVE:

#           best = mae

#           count += 1

#           gp.set_params(generations=count, warm_start=True)

#           imp_count = 0

#           best_iter = run

#           if mae < MAE_THRESH:

#               best_mdl = copy.deepcopy(mdl)

#       elif imp_count < MAX_NO_IMPROVE:

#           count += 1

#           gp.set_params(generations=count, warm_start=True)

#           imp_count += 1

#       else:

#           break



#       print('GP MAE: %.4f, Run: %d, Best Run: %d, Fold: %d' % (mae, run, best_iter, fold_))



#   maes.append(best)

#   gens.append(run)

      

#   print('Finish - GP MAE: %.4f, Run: %d, Best Run: %d' % (mae, run, best_iter))

          

#   preds = best_mdl.predict(scaled_test_X)

#   print(preds[0:12])

#   predictions += preds / folds.n_splits



# try:

#     print(maes)

#     print(np.mean(maes))

#     print(gens)

# except:

#     print('oops')

# submission.time_to_failure = predictions

# submission.to_csv('submission_gplearn_AF0_1.csv')

# print(submission.head(12))

print('ok')