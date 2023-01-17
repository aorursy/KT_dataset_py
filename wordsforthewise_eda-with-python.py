import re

import os



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



# "magic" command to make plots show up in the notebook

%matplotlib inline 



filepath = '../input/'
folders = os.listdir(filepath)

# Skip .xslx file, if we are ever able to upload it...

# Currently can't upload due to conflicts with other versions of the dataset on Kaggle.

folders = [f for f in folders if 'xlsx' not in f]

folders
os.listdir(filepath + folders[0])
acc_folder = filepath + [f for f in folders if 'accepted' in f][0]

accepted_fn = acc_folder + '/' + os.listdir(acc_folder)[0]



rej_folder = filepath + [f for f in folders if 'rejected' in f][0]

rejected_fn = rej_folder + '/' + os.listdir(rej_folder)[0]



accepted_fn
if os.path.isfile(accepted_fn) and os.path.isfile(rejected_fn):

    print('both paths still point to the actual file; all is good')

else:

    print('Kaggle changed how they handle compressed files again...you need to locate the files')
# Takes a while to read, because these files are large...give it a minute or so

acc_df = pd.read_csv(accepted_fn)



# this is a dataset with rejected loans from lendingclub

rej_df = pd.read_csv(rejected_fn)
# this is how we can see how many entries are in the dataframe (df)

# it also works for numpy arrays

acc_df.shape
rej_df.shape
[col for col in acc_df.columns if 'fico' in col.lower()]
# No fico score in rejected loans

[col for col in rej_df.columns if 'fico' in col.lower()]
rej_df.info()
acc_df.info()
# this is how many rows pandas will show by default with methods like pd.dataframe.head()

pd.options.display.max_rows
# we want to increase it, because in this case there are a lot of column names

pd.options.display.max_rows = 1000
# the .T is transposing the matrix.

# We do this so the 111 column dataframe is easier to read (easier to scroll down than sideways) 

# .head() shows the first few rows of the data

acc_df.head().T
# .tail() shows the last few rows

acc_df.tail().T
# .info() tells us the datatype(int64, `object` is a string)

# and will also tell us the number of non-null (not missing) data points for each column

# because this dataframe is so large, we have to force it to show the datatypes and non-null numbers with the arguments

acc_df.info(verbose=True, null_counts=True)
# shows some common summary statistics

# again, transposing with .T to make it easier to read

acc_df.describe().T
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import re

# "magic" command to make plots show up in the notebook

%matplotlib inline 
acc_df[acc_df['int_rate'] > 20]['int_rate'].mean()

# thats a bit of a complicated statement.  Breaking it down:

# acc_df['int_rate'] > 20 returns a mask: an array of True/False values

# putting this array into acc_df[] returns the dataframe rows where the interest rates are greater than 20



# so this first part: acc_df[acc_df['int_rate'] > 20]

# gives us a dataframe



# we select a column with ['int_rate'] at the end.  Then we get the average value with .mean()
acc_df['grade'].unique()
# selecting only grade A loans:

acc_df[acc_df['grade'] == 'A'].describe().T
acc_df['loan_status'].unique()
# looking at only defaulted loans:

default_categories = ['Default', 'Charged Off', 'Does not meet the credit policy. Status:Charged Off']

# .isin() is a trick for checking if something is in a list

# it's a pandas-specific function

acc_df[acc_df['loan_status'].isin(default_categories)].describe().T

# check out the average interest rate and dti (debt-to-income)
# Seaborn is a recently-created Python library for easily making nice-looking plots

# You will have to install it with `conda install seaborn` or `pip install seaborn`, etc



# The docs are here for this function: http://seaborn.pydata.org/generated/seaborn.distplot.html

# Found by Googling 'seaborn histogram'

# Need to drop missing values, otherwise throws an error

# It would be better to impute missing values.

f = sns.distplot(acc_df['dti'].dropna())
# outliers are screwing up the histogram... remove them

# adapted from http://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list

# we're using interquartile range to determine outliers

def reject_outliers(sr, iq_range=0.5, side='left', return_mask=False):

    """

    Takes an array (or pandas series) and returns an array with outliers excluded, according to the

    interquartile range.

    

    Parameters:

    -----------

    sr: array

        array of numeric values

    iq_range: float

        percent to calculate quartiles by, 0.5 will yield 25% and 75%ile quartiles

    side: string

        if 'left', will return everything below the highest quartile

        if 'right', will return everything above the lowest quartile

        if 'both', will return everything between the high and low quartiles

    """

    pcnt = (1 - iq_range) / 2

    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])

    iqr = qhigh - qlow

    if side=='both':

        mask = (sr - median).abs() <= iqr

    elif side=='left':

        mask = (sr - median) <= iqr

    elif side=='right':

        mask = (sr - median) >= iqr

    else:

        print('options for side are left, right, or both')

    

    if return_mask:

        return mask

    

    return sr[mask]
# sweeeeeeeetttt....

dti_no_outliers = reject_outliers(acc_df['dti'], iq_range=0.85) # arrived at 0.85 via trial and error

f = sns.distplot(dti_no_outliers)

# other types of plot examples:

# http://seaborn.pydata.org/examples/
# sets the xkcd style if you want to make the plots look funny...may neet to install some fonts

plt.xkcd()
f = sns.distplot(dti_no_outliers)
# back to default

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
f = sns.distplot(dti_no_outliers)
# other styles:

# http://matplotlib.org/users/style_sheets.html

# this is the R ggplot style, which some people really like

plt.style.use('ggplot')
# http://seaborn.pydata.org/generated/seaborn.regplot.html

# takes a long time because there are a lot of points, but works for smaller datasets

# f = sns.regplot(data=acc_df, x='dti', y='int_rate', fit_reg=False)

# instead, lets make a heatmap:

# http://stackoverflow.com/questions/2369492/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set

# http://seaborn.pydata.org/generated/seaborn.jointplot.html

mask = reject_outliers(acc_df['dti'], iq_range=0.85, return_mask=True)

f = sns.jointplot(data=acc_df.ix[mask, :], x='dti', y='int_rate', kind='hex', joint_kws=dict(gridsize=50))