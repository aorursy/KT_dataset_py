# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
sub1 = pd.read_csv('../input/ieee-blend/lgb_sub.csv')

sub2 = pd.read_csv('../input/ieee-blend/submission_IEEE (1).csv')

sub3 = pd.read_csv('../input/ieee-blend/submission_IEEE.csv')

sub4 = pd.read_csv('../input/ieee-blend/submission.csv')

temp=pd.read_csv('../input/ieee-blend/lgb_sub.csv')
sns.set()

plt.hist(sub1['isFraud'],bins=100)

plt.show()
sns.set()

plt.hist(sub2['isFraud'],bins=100)

plt.show()
sns.set()

plt.hist(sub3['isFraud'],bins=100)

plt.show()
sns.set()

plt.hist(sub4['isFraud'],bins=100)

plt.show()
#temp['isFraud'] = 0.35*sub1['isFraud'] + 0.30*sub2['isFraud'] + 0.25*sub3['isFraud'] + 0.10*sub4['isFraud'] 

#temp.to_csv('submission8.csv', index=False )
temp['isFraud'] = 0.60*sub4['isFraud'] + 0.40*sub3['isFraud']

temp.to_csv('submission_p2_1.csv', index=False )
sub_path = "../input/ieee-blend"

all_files = os.listdir(sub_path)

all_files
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "ieee" + str(x), range(len(concat_sub.columns))))

concat_sub.columns = cols

concat_sub.reset_index(inplace=True)

concat_sub.head()

ncol = concat_sub.shape[1]
# check correlation

concat_sub.iloc[:,1:ncol].corr()
corr = concat_sub.iloc[:,1:7].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# get the data fields ready for stacking

concat_sub['ieee_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)

concat_sub['ieee_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)

concat_sub['ieee_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)

concat_sub['ieee_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)
concat_sub.describe()
cutoff_lo = 0.7

cutoff_hi = 0.3
concat_sub['isFraud'] = concat_sub['ieee_mean']

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['isFraud'] = concat_sub['ieee_median']

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_median.csv', 

                                        index=False, float_format='%.6f')

concat_sub['isFraud'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 1, 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             0, concat_sub['ieee_median']))

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_pushout_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['isFraud'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['ieee_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['ieee_min'], 

                                             concat_sub['ieee_mean']))

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_minmax_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['isFraud'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['ieee_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['ieee_min'], 

                                             concat_sub['ieee_median']))

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_minmax_median.csv', 

                                        index=False, float_format='%.6f')
sub_base = pd.read_csv('../input/ieee-blend/lgb_sub.csv')
concat_sub['ieee_base'] = sub_base['isFraud']

concat_sub['isFraud'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['ieee_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['ieee_min'], 

                                             concat_sub['ieee_base']))

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_minmax_bestbase.csv', 

                                        index=False, float_format='%.6f')