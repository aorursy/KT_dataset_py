# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.datasets import load_diabetes

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

from IPython.display import Image



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
diabetes = load_diabetes()

X, y = diabetes.data, diabetes.target

n_features = X.shape[1]



# gradient boosted trees tend to do well on problems like this

reg = GradientBoostingRegressor(n_estimators=50, random_state=0)
from skopt.space import Real, Integer

from skopt.utils import use_named_args





# The list of hyper-parameters we want to optimize. For each one we define the bounds,

# the corresponding scikit-learn parameter name, as well as how to sample values

# from that dimension (`'log-uniform'` for the learning rate)

space  = [Integer(1, 5, name='max_depth'),

          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),

          Integer(1, n_features, name='max_features'),

          Integer(2, 100, name='min_samples_split'),

          Integer(1, 100, name='min_samples_leaf')]



# this decorator allows your objective function to receive a the parameters as

# keyword arguments. This is particularly convenient when you want to set scikit-learn

# estimator parameters

@use_named_args(space)

def objective(**params):

    reg.set_params(**params)



    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,

                                    scoring="neg_mean_absolute_error"))
from skopt import gp_minimize

import numpy as np

res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)



"Best score=%.4f" % res_gp.fun
print("""Best parameters:

- max_depth=%d

- learning_rate=%.6f

- max_features=%d

- min_samples_split=%d

- min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1], 

                            res_gp.x[2], res_gp.x[3], 

                            res_gp.x[4]))
from skopt.plots import plot_convergence



plot_convergence(res_gp);