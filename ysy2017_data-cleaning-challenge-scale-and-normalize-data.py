# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
kickstarters_2017 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)
np.random.exponential(size = 20)
fig, ax=plt.subplots(1,2)
ax[1]
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")
kickstarters_2017.usd_goal_real.head()
# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
scaled_data
# Your turn! 

# We just scaled the "usd_goal_real" column. What about the "goal" column?
kickstarters_2017.head()
# %load /opt/conda/lib/python3.6/site-packages/mlxtend/preprocessing/scaling.py
# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Classes for column-based scaling of datasets
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import pandas as pd
import numpy as np


def minmax_scaling(array, columns, min_val=0, max_val=1):
    """Min max scaling of pandas' DataFrames.

    Parameters
    ----------
    array : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].
    columns : array-like, shape = [n_columns]
        Array-like with column names, e.g., ['col1', 'col2', ...]
        or column indices [0, 2, 4, ...]
    min_val : `int` or `float`, optional (default=`0`)
        minimum value after rescaling.
    min_val : `int` or `float`, optional (default=`1`)
        maximum value after rescaling.

    Returns
    ----------
    df_new : pandas DataFrame object.
        Copy of the array or DataFrame with rescaled columns.

    """
    ary_new = array.astype(float)
    if len(ary_new.shape) == 1:
        ary_new = ary_new[:, np.newaxis]

    if isinstance(ary_new, pd.DataFrame):
        ary_newt = ary_new.loc
    elif isinstance(ary_new, np.ndarray):
        ary_newt = ary_new
    else:
        raise AttributeError('Input array must be a pandas'
                             'DataFrame or NumPy array')

    numerator = ary_newt[:, columns] - ary_newt[:, columns].min(axis=0)
    denominator = (ary_newt[:, columns].max(axis=0) -
                   ary_newt[:, columns].min(axis=0))
    ary_newt[:, columns] = numerator / denominator

    if not min_val == 0 and not max_val == 1:
        ary_newt[:, columns] = (ary_newt[:, columns] *
                                (max_val - min_val) + min_val)

    return ary_newt[:, columns]


def standardize(array, columns=None, ddof=0, return_params=False, params=None):
    """Standardize columns in pandas DataFrames.

    Parameters
    ----------
    array : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].
    columns : array-like, shape = [n_columns] (default: None)
        Array-like with column names, e.g., ['col1', 'col2', ...]
        or column indices [0, 2, 4, ...]
        If None, standardizes all columns.
    ddof : int (default: 0)
        Delta Degrees of Freedom. The divisor used in calculations
        is N - ddof, where N represents the number of elements.
    return_params : dict (default: False)
        If set to True, a dictionary is returned in addition to the
        standardized array. The parameter dictionary contains the
        column means ('avgs') and standard deviations ('stds') of
        the individual columns.
    params : dict (default: None)
        A dictionary with column means and standard deviations as
        returned by the `standardize` function if `return_params`
        was set to True. If a `params` dictionary is provided, the
        `standardize` function will use these instead of computing
        them from the current array.

    Notes
    ----------
    If all values in a given column are the same, these values are all
    set to `0.0`. The standard deviation in the `parameters` dictionary
    is consequently set to `1.0` to avoid dividing by zero.

    Returns
    ----------
    df_new : pandas DataFrame object.
        Copy of the array or DataFrame with standardized columns.

    """
    ary_new = array.astype(float)
    dim = ary_new.shape
    if len(dim) == 1:
        ary_new = ary_new[:, np.newaxis]

    if isinstance(ary_new, pd.DataFrame):
        ary_newt = ary_new.loc
        if columns is None:
            columns = ary_new.columns
    elif isinstance(ary_new, np.ndarray):
        ary_newt = ary_new
        if columns is None:
            columns = list(range(ary_new.shape[1]))

    else:
        raise AttributeError('Input array must be a pandas '
                             'DataFrame or NumPy array')

    if params is not None:
        parameters = params
    else:
        parameters = {'avgs': ary_newt[:, columns].mean(axis=0),
                      'stds': ary_newt[:, columns].std(axis=0, ddof=ddof)}
    are_constant = np.all(ary_newt[:, columns] == ary_newt[0, columns], axis=0)

    for c, b in zip(columns, are_constant):
        if b:
            ary_newt[:, c] = np.zeros(dim[0])
            parameters['stds'][c] = 1.0

    ary_newt[:, columns] = ((ary_newt[:, columns] - parameters['avgs']) /
                            parameters['stds'])

    if return_params:
        return ary_newt[:, columns], parameters
    else:
        return ary_newt[:, columns]

goal = kickstarters_2017.goal
goal_scaled = minmax_scaling(goal, columns=[0])
fig, ax = plt.subplots(1,2)
sns.distplot(goal, ax=ax[0])
sns.distplot(goal_scaled, ax=ax[1])
# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")
