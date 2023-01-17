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


# function to return growth
# later will be used to calculate year-on-year growth
def growth(df, date_col='date', count_col='count', resample_period='AS'):
    df_ri = df.set_index(date_col)
    df_resample = df_ri[count_col].resample(resample_period).sum().to_frame().reset_index()

    df_resample['prev'] = df_resample[date_col] - pd.DateOffset(years=1)

    growth_df = df_resample.merge(
        df_resample,
        how='inner',
        left_on=[date_col],
        right_on=['prev'],
        suffixes=['_l', '_r']
    ).loc[:,['{}_r'.format(date_col), '{}_l'.format(count_col), '{}_r'.format(count_col)]].rename(
        columns={
            '{}_r'.format(date_col) : 'date',
            '{}_l'.format(count_col) : 'count_last_year',
            '{}_r'.format(count_col) : 'count_current'
        }
    )
    
    growth_df['growth'] = growth_df['count_current'] / growth_df['count_last_year']
    return growth_df
# load data
usr_df = pd.read_csv("../input/yelp_user.csv")
rvw_df = pd.read_csv("../input/yelp_review.csv")

# data type conversion
usr_df.yelping_since = pd.to_datetime(usr_df.yelping_since)
rvw_df.date = pd.to_datetime(rvw_df.date)

# check what is inside dataframe
print('This is the user dataframe columns:')
print(usr_df.dtypes)
print()
print('This is the review dataframe columns:')
print(rvw_df.dtypes)
# How many user in the dataset?
'There are {:,} users'.format(len(usr_df))
# if there are users with same user_id
count_user_id = usr_df[['user_id','name']].groupby(['user_id']).count().rename(columns={'name' : 'count'})
assert (len(count_user_id[count_user_id['count'] > 1]) == 0), "Multiple user with same user_id"
count_yelping_since = usr_df[['yelping_since', 'user_id']].groupby(['yelping_since']).count().rename(columns={'user_id':'count'}).resample('1d').sum().fillna(0)
count_yelping_since['rolling_mean_30d'] = count_yelping_since.rolling(window=30, min_periods=1).mean()
count_yelping_since = count_yelping_since.reset_index()

fig, ax = plt.subplots(figsize=(12,7.5))

_ = count_yelping_since.plot(
    ax=ax,
    x='yelping_since', 
    y='rolling_mean_30d'
)

_ = count_yelping_since.plot(
    ax=ax,
    x='yelping_since', 
    y='count',
    alpha=.3
)

_ = ax.set_title('Daily Number of New User')
_ = ax.legend(['Rolling Mean 30 Days', 'Daily Count'])
_ = ax.set_yticklabels(['{:,.0f}'.format(x) for x in ax.get_yticks()])
yoy_user_df = growth(count_yelping_since, date_col='yelping_since')
ax1 = yoy_user_df.plot(
    x='date', 
    y='growth', 
    title='Overall Year-on-Year User Growth', 
    figsize=(7.5,5), 
    legend=False,
    linewidth=3
)
_ = ax1.set_yticklabels(['{:,.0f}%'.format(x*100) for x in ax1.get_yticks()])
ax2 = yoy_user_df[yoy_user_df['date'] >= '2008-01-01'].plot(
    x='date', 
    y='growth', 
    title='Year-on-Year User Growth After 2008', 
    figsize=(12,7.5),
    legend=False,
    linewidth=3
)
_ = ax2.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax2.get_yticks()])
# How many user in the dataset?
'There are {:,} reviews'.format(len(rvw_df))
#count_review = rvw_df[['date', 'review_id']].groupby(['date']).count().rename(columns={'review_id':'count'})
count_review = rvw_df[['date', 'review_id']].groupby(['date']).count().rename(columns={'review_id':'count'}).resample('1d').sum().fillna(0)
count_review['rolling_mean_30d'] = count_review.rolling(window=30, min_periods=1).mean()
count_review = count_review.reset_index()

fig, ax = plt.subplots(figsize=(12,7.5))

_ = count_review.plot(
    ax=ax,
    x='date', 
    y='rolling_mean_30d',
    style='--'
)

_ = count_review.plot(
    ax=ax,
    x='date', 
    y='count',
    alpha=.3
)

_ = ax.set_title('Daily Number of New Review')
_ = ax.legend(['Rolling Mean 30 Days', 'Daily Count'])
_ = ax.set_yticklabels(['{:,.0f}'.format(x) for x in ax.get_yticks()])
yoy_review = growth(count_review)

fig, ax = plt.subplots(figsize=(12,7.5))

_ = yoy_review[yoy_review['date'] >= '2008-01-01'].plot(
    ax=ax,
    x='date', 
    y='growth',
    linewidth=3
)

_ = yoy_user_df[yoy_user_df['date'] >= '2008-01-01'].plot(
    ax=ax,
    x='date', 
    y='growth',
    style='--',
    linewidth=3
)

_ = ax.set_title('Yearly User and Review Growth Comparison')
_ = ax.legend(['#Review Growth', '#User Growth'])
_ = ax.set_yticklabels(['{:,.0f}%'.format(x*100) for x in ax.get_yticks()])
# count monthly review per user with review_count and yelping_since
usr_rvw = usr_df[['user_id', 'review_count', 'yelping_since']].copy(deep=True)

# get latest date in dataset,
# safe check, if latest yelping since > latest review date, then take latest yelping since instead
# add buffer 1 month so there is no 0 'month since'
latest_date = max(rvw_df['date'].max(), usr_rvw['yelping_since'].max()) + np.timedelta64(1, 'M') 
usr_rvw['month_since'] = (latest_date - usr_rvw['yelping_since']) / np.timedelta64(1, 'M')
usr_rvw['review_per_month'] = usr_rvw['review_count'] / usr_rvw['month_since']
ax = usr_rvw['review_per_month'].plot.hist(figsize=(7.5,5))
_ = ax.set_yticklabels(['{:,.2f}m'.format(x/1000000) for x in ax.get_yticks()])
_ = ax.set_xlabel('monthly number of review')
_ = ax.set_title('Number of Review per User Distribution')
usr_rvw_q = usr_rvw.review_per_month.quantile([.9, .95, .99, 1])
print(usr_rvw_q)
# We cut at 99% to remove outliers
usr_rvw_rem_outliers = usr_rvw[usr_rvw['review_per_month'] <= usr_rvw_q.loc[0.99]]['review_per_month']
weights = np.ones_like(usr_rvw_rem_outliers) / float(len(usr_rvw_rem_outliers))

ax = usr_rvw_rem_outliers.plot.hist(bins=int(usr_rvw_q.loc[0.99] * 8), weights=weights, figsize=(12,9))
_ = ax.set_xlabel('monthly number of review')
_ = ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax.get_yticks()])
_ = ax.set_title('Number of Review per User Distribution')
'{:.5f}% of users never review'.format(len(usr_rvw[usr_rvw['review_per_month'] == 0]) * 100 / len(usr_rvw[usr_rvw['review_per_month'] < 4]))
rvw_eng = rvw_df[['review_id', 'user_id', 'date']].copy(deep=True)
rvw_eng['year'] = rvw_eng['date'].map(lambda x: x.year)

# prepare user dataframe
# to accomodate user who never review
usr_rvw['year_join'] = usr_rvw['yelping_since'].map(lambda x: x.year)
# find out for each year, what is the distribution of monthly number of review per user

years = rvw_eng['year'].unique()
yearly_rvw_df = pd.DataFrame(columns=['user_id', 'year', 'rvw_p_month'])

for year in years:
    # get all the users that exist this year
    usr_prev_year = usr_rvw[usr_rvw['year_join'] < year][['user_id', 'yelping_since']]
    usr_prev_year['yearly_month_since'] = 12.0 # this means user has joined for the full year
    
    usr_curr_year = usr_rvw[usr_rvw['year_join'] == year][['user_id', 'yelping_since']]
    usr_curr_year['yearly_month_since'] = (pd.Timestamp('{}-01-01'.format(year+1)) - usr_curr_year['yelping_since']) / np.timedelta64(1, 'M')
    
    usr_curr_year_all = usr_curr_year.append(usr_prev_year)
    
    # now get all review done in current year and count by user
    rvw_curr_year = rvw_eng[rvw_eng['year'] == year][['user_id', 'review_id']].groupby(['user_id']).count().rename(columns={'review_id':'count'}).reset_index()
    
    usr_curr_year_all = usr_curr_year_all.merge(rvw_curr_year, on='user_id', how='left')
    usr_curr_year_all['count'].fillna(0.0, inplace=True)
    usr_curr_year_all['rvw_p_month'] = usr_curr_year_all['count'] / usr_curr_year_all['yearly_month_since']
    usr_curr_year_all['year'] = year
    
    yearly_rvw_df = yearly_rvw_df.append(usr_curr_year_all[['user_id', 'year', 'rvw_p_month']])

yearly_rvw_df['year'] = pd.to_numeric(yearly_rvw_df['year'])
         
no_review_proportion = (yearly_rvw_df[yearly_rvw_df['rvw_p_month'] == 0].groupby(['year']).count().rvw_p_month / yearly_rvw_df.groupby(['year']).count().rvw_p_month).to_frame().reset_index()

yoy_review['year'] = yoy_review.apply(lambda x: x['date'].year, axis=1)

fig, ax = plt.subplots(figsize=(12,7.5))

_ = no_review_proportion[no_review_proportion['year'] >= 2008].plot(
    ax=ax,
    x='year',
    y='rvw_p_month',
    linewidth=2
)

_ = yoy_review[yoy_review['year'] >= 2008].plot(
    ax=ax,
    x='year', 
    y='growth',
    linewidth=2,
    style='--'
)

_ = ax.set_title('Correlation Between User Who Writes No Review to User Growth')
_ = ax.legend(['Proportion of User Who Writes No Review', '#User Growth'])
_ = ax.set_yticklabels(['{:2.0f}%'.format(x*100) for x in ax.get_yticks()])

# We cut at 99% to remove outliers
yearly_rvw_median = yearly_rvw_df[yearly_rvw_df['rvw_p_month'] > 0].groupby('year')['rvw_p_month'].quantile(.5)
ax = yearly_rvw_median.plot.bar(figsize=(12,9), title='Monthly Number of Review per User')
_ = ax.set_ylabel('median (review/month)')