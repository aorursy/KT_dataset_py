import numpy as np 

import pandas as pd

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn import linear_model

import matplotlib.pyplot as plt

df= pd.read_csv('../input/patient-satisfaction/satis_data.csv') 

print(df)
# calculate the correlation matrix

corr = df.corr()



# display the correlation matrix

display(corr)



# plot the correlation heatmap

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu')
import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt



import statsmodels.api as sm

from statsmodels.stats import diagnostic as diag

from statsmodels.stats.outliers_influence import variance_inflation_factor



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



%matplotlib inline
df.head()
# get the summary

df = df.describe()



# add the standard deviation metric

df.loc['+3_std'] = df.loc['mean'] + (df.loc['std'] * 3)

df.loc['-3_std'] = df.loc['mean'] - (df.loc['std'] * 3)



# display it

df
# define the plot

pd.plotting.scatter_matrix(df, alpha = 1, figsize = (20, 50))



# show the plot

plt.show()