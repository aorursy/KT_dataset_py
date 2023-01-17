%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# options
pd.set_option('display.max_colwidth', -1)

# extra config to have better visualization
sns.set(
    style='whitegrid',
    palette='coolwarm',
    rc={'grid.color' : '.96'}
)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.figsize'] = 12, 7.5
# load data
rvw_df = pd.read_csv("../input/yelp_review.csv")
biz_df = pd.read_csv("../input/yelp_business.csv")

# check what is inside dataframe
print('This is the review dataframe columns:')
print(rvw_df.dtypes)
rvw_df['funny'].describe()
zero_funny = len(rvw_df[rvw_df['funny'] == 0])
zero_funny_prop = (zero_funny * 100) / len(rvw_df)
'There are {:,d} ({:2.3f}% of overall) reviews with 0 funny votes'.format(zero_funny, zero_funny_prop)
print(rvw_df['text'].values[rvw_df['funny'] == max(rvw_df['funny'])][0])
rvw_df[['business_id', 'funny']].sort_values(by='funny', ascending=False).head(10)
rvw_df[['business_id', 'funny']].sort_values(by='funny', ascending=False).head(50)
biz_df[biz_df['business_id'] == 'DN0b4Un8--Uf6SEWLeh0UA']
rvw_df[(rvw_df['funny'] > 0) & (rvw_df['business_id'] != 'DN0b4Un8--Uf6SEWLeh0UA')]['funny'].quantile([.95, .99, .999, 1])
print(rvw_df['text'].values[rvw_df['funny'] == 216][0])
ax = rvw_df[(rvw_df.funny > 0) & (rvw_df.funny <= 16)]['funny'].hist(
    bins=range(1, 16, 1), 
    normed=True
)

_ = ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax.get_yticks()])
_ = ax.set_title('Normalized Distribution for Funny Votes')
_ = ax.set_xlabel('Number of Funny Votes')
'There are {:,d} reviews with 16 votes'.format(len(rvw_df[rvw_df['funny'] == 16]))
rvw_df[['text','useful','cool']][rvw_df['funny'] == 16].head(3)
rvw_df[rvw_df['useful'] > 0]['useful'].quantile(.999)
rvw_df[rvw_df['cool'] > 0]['cool'].quantile(.999)
rvw_fun_df = rvw_df[((rvw_df['funny'] > 0) | (rvw_df['useful'] > 0) | (rvw_df['cool'] > 0))
                    & (rvw_df['funny'] <= 50) 
                    & (rvw_df['useful'] <= 50) 
                    & (rvw_df['cool'] <= 50) 
                    & (rvw_df['business_id'] != 'DN0b4Un8--Uf6SEWLeh0UA')
                   ].reset_index()[['funny','useful','cool']].copy(deep=True)
'There are {:,d} reviews that fit the criteria'.format(len(rvw_fun_df))
for c in ['funny', 'useful', 'cool']:
    jitter = np.random.normal(0, 0.002, size=len(rvw_fun_df))
    rvw_fun_df['z_' + c] = ((rvw_fun_df[c] - rvw_fun_df[c].mean()) / (rvw_fun_df[c].max() - rvw_fun_df[c].min())) + jitter
rvw_fun_df.head(5)
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12,20))

_ = rvw_fun_df.plot.scatter(
    ax=ax[0],
    x='z_funny',
    y='z_useful',
    color='indianred',
    alpha=.1
)

_ = rvw_fun_df.plot.scatter(
    ax=ax[1],
    x='z_funny',
    y='z_cool',
    color='violet',
    alpha=.1
)

_ = rvw_fun_df.plot.scatter(
    ax=ax[2],
    x='z_cool',
    y='z_useful',
    color='hotpink',
    alpha=.1
)
rvw_fun_df[['funny','useful','cool']].corr()
