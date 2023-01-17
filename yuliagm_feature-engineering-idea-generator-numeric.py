import numpy as np 
import pandas as pd 
import datetime
import os
import gc

import time

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }

columns = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'click_time']
train_smp = pd.read_csv('../input/train.csv', dtype=dtypes, nrows=10000000, parse_dates=['click_time'], usecols=columns  )
train_smp.head()
################# HELPER AGGREGATION FUNCTIONS ################
#total count features
def do_count( df, group_cols, agg_name, agg_type='uint16', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

#unique count features
def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint16', show_max=False, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
    
#cummulative count features    
def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint16', show_max=False, show_agg=True ):
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )


######## Data Preprocessing  ####
def prep_data(df):
    
    print('Extracting new features...')
    
    #time prep
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df['min'] = pd.to_datetime(df.click_time).dt.minute.astype('uint8')

    ## UNIQUE COUNTS FEATURES:
    df = do_countuniq( df, ['ip'], 'app', 'uq_app_per_ip', show_max=True ); gc.collect()
    df = do_countuniq( df, ['ip', 'app'], 'os', 'uq_os_per_ip_app', show_max=True ); gc.collect()
    df = do_countuniq( df, ['ip'], 'device', 'uq_device_per_ip', show_max=True ); gc.collect()
    df = do_countuniq( df, ['app'], 'channel', 'uq_channels_per_app', show_max=True ); gc.collect()
    df = do_countuniq( df, ['ip'], 'channel', 'uq_channel_per_ip', show_max=True ); gc.collect()

    ## TOTALS COUNTS FEATURES:
    df = do_count( df, ['ip', 'day'], 'ip_per_day_count', show_max=True ); gc.collect()
    df = do_count( df, ['ip', 'app'], 'ip_app_count', show_max=True ); gc.collect()
    df = do_count( df, ['ip', 'app', 'os'], 'ip_app_os_count', show_max=True ); gc.collect()
    df = do_count( df, ['ip', 'channel', 'hour'], 'ip_channel_per_hour', show_max=True ); gc.collect()

    ##### CUMMULATIVE COUNTS FEATURES:
    df = do_cumcount( df, ['ip', 'device', 'os'], 'app', 'cumcount_ip_dev_os', show_max=True ); gc.collect()

    print('*'*30)
    print('after data prep:')
    print(df.head())
    print('finished feature generation')
    
    return (df)
#create new features
train_smp = prep_data(train_smp)
unique_count_features = ['uq_app_per_ip', 'uq_os_per_ip_app', 'uq_device_per_ip', 'uq_channels_per_app', 'uq_channel_per_ip']

other_count_features = ['ip_per_day_count', 'ip_app_count', 'ip_app_os_count', 'ip_channel_per_hour', 'cumcount_ip_dev_os']
#make wider graphs
sns.set(rc={'figure.figsize':(12,5)});
plt.figure(figsize=(12,5));

for fea in unique_count_features:
    sns.boxplot(train_smp.is_attributed, train_smp[fea])
    title_2 = 'PLOT OF: ' +fea
    plt.title(title_2)
    plt.show()
    sns.violinplot(train_smp.is_attributed, train_smp[fea])
    title_2 = 'PLOT OF:  ' +fea
    plt.title(title_2)
    plt.show()
    gc.collect()
    print('*'*70)

#visualize distribution by remaining count features, normal boxplot, log boxplot and log violinplot
for fea in other_count_features:
    sns.boxplot(train_smp.is_attributed, train_smp[fea])
    title = 'REGULAR BOXPLOT PLOT OF: ' +fea
    plt.title(title)
    plt.show()
    sns.boxplot(train_smp.is_attributed, np.log(train_smp[fea]+1))
    title = 'LOG BOXPLOT PLOT OF: ' +fea
    plt.title(title)
    plt.show()
    sns.violinplot(train_smp.is_attributed, np.log(train_smp[fea]+1))
    title = 'LOG VIOLINPLOT PLOT OF:  ' +fea
    plt.title(title_2)
    plt.show()
    gc.collect()
    print('*'*70)

#visualize distribution of attributions by minute
sns.boxplot(train_smp.is_attributed, train_smp['min'])
plt.title('Boxplot of Minute distribution')
plt.show()
sns.violinplot(train_smp.is_attributed, train_smp['min'])
plt.title('Violinplot of Minute distribution')
plt.show()
#generate next_click feature
train_smp['click_time'] = (train_smp['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
train_smp['next_click'] = (train_smp.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - train_smp.click_time).astype(np.float32)
train_smp['next_click'] = train_smp['next_click'].fillna(360000000).astype('uint32')
#visualize next_click
sns.boxplot(train_smp.is_attributed, np.log(train_smp['next_click']+1))
plt.title('LOG Boxplot of Next_Click')
plt.show()
sns.violinplot(train_smp.is_attributed, np.log(train_smp['next_click']+1))
plt.title('LOG Violinplot of Next_click')
plt.show()
