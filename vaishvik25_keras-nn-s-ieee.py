import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
sub_path = "../input/top-mol"

all_files = os.listdir(sub_path)

all_files
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))

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

concat_sub['m_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)

concat_sub['m_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)

concat_sub['m_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)

concat_sub['m_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)
concat_sub.describe()
cutoff_lo = 0.8

cutoff_hi = 0.2
concat_sub['scalar_coupling_constant'] = concat_sub['m_mean']

concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['scalar_coupling_constant'] = concat_sub['m_median']

concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['scalar_coupling_constant'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 1, 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             0, concat_sub['m_median']))

concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_pushout_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['scalar_coupling_constant'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['m_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['m_min'], 

                                             concat_sub['m_mean']))

concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_minmax_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['scalar_coupling_constant'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['m_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['m_min'], 

                                             concat_sub['m_median']))

concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_minmax_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['scalar_coupling_constant'] = concat_sub['mol0'].rank(method ='min') + concat_sub['mol1'].rank(method ='min') + concat_sub['mol2'].rank(method ='min') 

concat_sub['scalar_coupling_constant'] = (concat_sub['scalar_coupling_constant']-concat_sub['scalar_coupling_constant'].min())/(concat_sub['scalar_coupling_constant'].max() - concat_sub['scalar_coupling_constant'].min())

concat_sub.describe()

concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_rank.csv', index=False, float_format='%.8f')
sub1=pd.read_csv('../input/another-one/stackers_blend.csv')

sub2=pd.read_csv('../input/lgb-public-kernels-plus-more-features/sub_lgb_model_individual.csv')

sub3=pd.read_csv('../input/yet-another-one/stackers_blend.csv')

sub4=pd.read_csv('../input/giba-r-data-table-simple-features-1-17-lb/submission-giba-1.csv')



temp=pd.read_csv('../input/another-one/stackers_blend.csv')
temp['scalar_coupling_constant'] = 0.6*sub1['scalar_coupling_constant'] + 0.4*sub1['scalar_coupling_constant']

temp.to_csv('submission4.csv', index=False )
sns.distplot(temp['scalar_coupling_constant'])