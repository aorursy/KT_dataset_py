#Our final submission is a single lightgbm model, no blending (due to limited computational power and time)
#We are going topresent all the pre_processing, feature engineering functions and lightgbm paramaters, enjoy
import pandas as pd
import numpy as np
import lightgbm as lgb
#import pytz
import gc
from sklearn import preprocessing
#all functions
def prepare(df):
    df['row_id'] = range(df.shape[0])
    df.click_time = pd.to_datetime(df.click_time)
    df['day'] = df.click_time.dt.day.astype('uint8')
    df['hour'] = df.click_time.dt.hour.astype('uint8')
    
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]
    
    df['in_test_hh'] = (   3 
                         - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    return df

def get_size(group_column, df):
    '''
    group_column: categorical columns to group on 
    '''
    
    used_cols = group_column.copy()
    used_cols.extend(['row_id'])
    all_df = df[used_cols]
    grouped = all_df[group_column].groupby(group_column)
    
    #size of each level in group_column
    the_size = pd.DataFrame(grouped.size()).reset_index()
    names = group_column.copy()
    new_name = "_".join(x for x in names) + '_size'
    names.append(new_name)
    the_size.columns = names
    
    all_df = pd.merge(all_df, the_size)
    all_df.sort_values('row_id', inplace=True)

    df[new_name] = np.array(all_df[new_name])
    del all_df
    gc.collect()
    
    return df

def get_unique(df, grouping_col, target_col):
    
    used_cols = grouping_col.copy()
    used_cols.extend(['row_id'])
    used_cols.extend(target_col)
    all_df = df[used_cols]
    
    group_used_cols = grouping_col.copy()
    group_used_cols.extend(target_col)
    grouped = all_df[group_used_cols].groupby(grouping_col)
    #unique count
    the_count = pd.DataFrame(grouped[target_col].nunique()).reset_index()
    names = grouping_col.copy()
    new_name = "_".join(x for x in target_col) + '_unique_count_on_' + "_".join(x for x in grouping_col)
    names.append(new_name)
    the_count.columns = names
    
    all_df = pd.merge(all_df, the_count)
    all_df.sort_values('row_id', inplace=True)

    df[new_name] = np.array(all_df[new_name])
    del all_df
    gc.collect()
    
    return df

def get_user_info(df):
    user = ['ip', 'device','os']
    #get total count
    df = get_size(user, df)
    new_name1 = "_".join(x for x in user) + '_size'
    #get total count on an app
    df = get_size(['ip', 'app', 'device','os'], df)
    new_name2 = "_".join(x for x in (['ip', 'app', 'device','os'])) + '_size'
    #get proportion of app count on this user
    new_name3 = new_name2 + '/' + new_name1
    df[new_name3] = df[new_name2]/df[new_name1]
    #get unique app count
    df = get_unique(df, user, ['app'])
    new_name4 = "_".join(x for x in ['app']) + '_unique_count_on_' + "_".join(x for x in user)
    #get unique/size ratio
    new_name5 = new_name4 + '/' + new_name1
    df[new_name5] = df[new_name4]/df[new_name1]
    #cumcount
    df['user_newness'] = df.groupby(user).cumcount() 
    df['user_app_oldness'] = df.groupby(user + ['app']).cumcount(ascending=False)
    df.drop(['ip_device_os_size'], axis=1, inplace=True)
    gc.collect()
    print('get_user_info done')
    return df

def get_kernel_fe(df):
    df = get_size(['ip','day','in_test_hh'], df)
    df = get_size(['ip','day','hour'], df)
    #df = get_size(['ip','day','hour','os'], df)
    df = get_size(['ip','day','hour','app'], df)
    df = get_size(['ip','day','hour','app','os'], df)
    print('get_kernel_fe half is done')
    df = get_size(['app','day','hour'], df)
    df = get_size(['ip','app'], df)
    df = get_size(['ip','app','os'], df)
    df = get_size(['ip','device'], df)
    df = get_size(['app','channel'], df)
    print('get_kernel_fe is done')
    gc.collect()
    return df    

def get_hourly_app_info(df):
    app = ['app', 'day','hour']
    df = get_unique(df, app, ['ip'])
    new_name1 = "_".join(x for x in ['ip']) + '_unique_count_on_' + "_".join(x for x in app)
    new_name2 = "_".join(x for x in app) + '_size'
    new_name3 = new_name1 + '/' + new_name2
    df[new_name3] = df[new_name1]/df[new_name2]
    
    df = get_size(['ip', 'app','device', 'os', 'day','hour'], df)
    gc.collect()
    print('get_hourly_app_info done')
    return df

def get_click_time_info(df):
    df.click_time = pd.to_datetime(df.click_time, errors = 'ignore')
    used_cols = ['ip', 'app', 'device', 'os', 'click_time', 'row_id']
    all_df = df[used_cols]
    all_df = all_df.sort_values(by=used_cols)
    
    all_df['next_ip']=all_df.ip.shift(-1)
    all_df['next_app']=all_df.app.shift(-1)
    all_df['next_device']=all_df.device.shift(-1)
    all_df['next_os']=all_df.os.shift(-1)
    all_df['next_click_time']=all_df.click_time.shift(-1)
    
    all_df['has_next_ip'] = np.where(all_df.ip == all_df.next_ip, 1, 0)
    all_df['has_next_app'] = np.where(all_df.app == all_df.next_app, 1, 0)
    all_df['has_next_device'] = np.where(all_df.device == all_df.next_device, 1, 0)
    all_df['has_next_os'] = np.where(all_df.os == all_df.next_os, 1, 0)
    
    all_df['next_click'] = np.where((all_df.has_next_ip == 1) & (all_df.has_next_app == 1) &(all_df.has_next_device == 1) & (all_df.has_next_os == 1) , (all_df.next_click_time - all_df.click_time)/np.timedelta64(1, 's'), np.NaN)
    
    
    all_df['previous_ip']=all_df.ip.shift(1)
    all_df['previous_app']=all_df.app.shift(1)
    all_df['previous_device']=all_df.device.shift(1)
    all_df['previous_os']=all_df.os.shift(1)
    all_df['previous_click_time']=all_df.click_time.shift(1)
    
    all_df['has_previous_ip'] = np.where(all_df.ip == all_df.previous_ip, 1, 0)
    all_df['has_previous_app'] = np.where(all_df.app == all_df.previous_app, 1, 0)
    all_df['has_previous_device'] = np.where(all_df.device == all_df.previous_device, 1, 0)
    all_df['has_previous_os'] = np.where(all_df.os == all_df.previous_os, 1, 0)
    
    all_df['last_click'] = np.where((all_df.has_previous_ip == 1) & (all_df.has_previous_app == 1) & (all_df.has_previous_device == 1) & (all_df.has_previous_os == 1) , (all_df.click_time-all_df.previous_click_time)/np.timedelta64(1, 's'), np.NaN)
    
    all_df = all_df.sort_values(by=['row_id'])
    df['next_click'] = np.array(all_df['next_click'])
    df['last_click'] = np.array(all_df['last_click'])
    del all_df
    gc.collect()
    print('get_click_time_info done')
    return df

def get_next_click_stat(df):
    grouping_col = ['ip','app','device','os']
    target_col = ['next_click']
    
    used_cols = grouping_col.copy()
    used_cols.extend(['row_id'])
    used_cols.extend(target_col)
    all_df = df[used_cols]
    
    group_used_cols = grouping_col.copy()
    group_used_cols.extend(target_col)
    grouped = all_df[group_used_cols].groupby(grouping_col)
    
    new_names = []
    #mean
    the_mean = pd.DataFrame(grouped[target_col].mean()).reset_index()
    names = grouping_col.copy()
    new_name = 'next_click_mean'
    new_names.append(new_name)
    names.append(new_name)
    the_mean.columns = names
    #median
    the_median = pd.DataFrame(grouped[target_col].median()).reset_index()
    names = grouping_col.copy()
    new_name = 'next_click_median'
    new_names.append(new_name)
    names.append(new_name)
    the_median.columns = names
    the_stats = pd.merge(the_mean, the_median)
    #max
    the_max = pd.DataFrame(grouped[target_col].max()).reset_index()
    names = grouping_col.copy()
    new_name = 'next_click_max'
    new_names.append(new_name)
    names.append(new_name)
    the_max.columns = names
    the_stats = pd.merge(the_stats, the_max)
    
    all_df = pd.merge(all_df, the_stats)
    all_df.sort_values('row_id', inplace=True)
    
    for new_name in new_names:
        df[new_name] = np.array(all_df[new_name])
    del all_df
    gc.collect()
    print('get_next_click_stat is done')
    return df

def get_old_size(df):
    candidates = [        
        ['ip', 'app', 'device','os'],
        ['app', 'device','os']
    ]
    df['minute'] = df.click_time.dt.minute.astype('uint8')
    df['second'] = df.click_time.dt.second.astype('uint8')
    gc.collect()
    for i in range(0, 2):
        used = candidates[i].copy()
        if i == 0:
            df = get_size(used + ['day', 'hour', 'minute'], df)
            df = get_size(used + ['day', 'hour', 'minute', 'second'], df)
            df['ip_app_device_os_size_hour/min_rate'] = df['ip_app_device_os_day_hour_size']/df['ip_app_device_os_day_hour_minute_size']
            df['ip_app_device_os_size_min/sec_rate'] = df['ip_app_device_os_day_hour_minute_size']/df['ip_app_device_os_day_hour_minute_second_size']
            dropped = ['ip_app_device_os_day_hour_minute_second_size']
            df.drop(dropped, axis=1, inplace=True)
            gc.collect()
        elif i == 1:
            df = get_size(used + ['day', 'hour'], df)
            df = get_size(used + ['day', 'hour', 'minute'], df)
            df = get_size(used + ['day', 'hour', 'minute', 'second'], df)
            df['app_device_os_size_hour/min_rate'] = df['app_device_os_day_hour_size']/df['app_device_os_day_hour_minute_size']
            df['app_device_os_size_hour/sec_rate'] = df['app_device_os_day_hour_size']/df['app_device_os_day_hour_minute_second_size']
            df['app_device_os_size_min/sec_rate'] = df['app_device_os_day_hour_minute_size']/df['app_device_os_day_hour_minute_second_size']
            dropped = ['app_device_os_day_hour_minute_size', 'app_device_os_day_hour_minute_second_size']
            df.drop(dropped, axis=1, inplace=True)
            gc.collect()
    df.drop(['minute','second'], axis=1, inplace=True)
    print('get old size done')
    gc.collect()
    return df

def get_old_unique(df):
    #os
    df = get_size(['ip'],df)
    df = get_unique(df, ['ip'], ['os'])
    new_name1 = 'os_unique_count_on_ip' + '/' + 'ip_size'
    df[new_name1] = df['os_unique_count_on_ip']/df['ip_size']
    #device
    df = get_unique(df, ['ip'], ['device'])
    new_name2 = 'device_unique_count_on_ip' + '/' + 'ip_size'
    df[new_name2] = df['device_unique_count_on_ip']/df['ip_size']
    
    df.drop(['ip_size'], axis=1, inplace=True)
    gc.collect()
    print('get old unique done')
    return df

def get_yulia_fe(df):
    df = get_unique(df, ['ip','device','os','day','hour'], ['channel'])
    df = get_size(['ip','channel'],df)
    print('get yulia fe is done')
    return df

def get_p1(train, num_rounds, if_ip=False):
    predictors = list(train.columns)
    remove_list = ['click_id', 'row_id', 'day', 'minute', 'second', 'click_time', 'local_click_time', 'attributed_time', 'is_attributed']
    for ele in remove_list:
        if ele in predictors:
            predictors.remove(ele)
    target = 'is_attributed'
    categorical = ['ip','app','os','device','channel','hour']
    if if_ip == False:
        predictors.remove('ip')
        categorical.remove('ip')
    params = {
        'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 'metric':'auc', 'seed':77,
        'num_leaves': 48, 'learning_rate': 0.01, 'max_depth': -1, 'gamma':47,
        'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction':0.7, 'bagging_freq':1, 'bagging_seed':55,
        'colsample_bytree': 0.6548, 'reg_alpha': 19.43, 'reg_lambda': 0, 
        'min_split_gain': 0.3512, 'min_child_weight': 0, 'min_child_samples':1321, 'scale_pos_weight':205}
    
    
    xgtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    bst = lgb.train(params, xgtrain, num_boost_round = num_rounds, verbose_eval=False)
    return bst, predictors
#the code you need to run to get the model
'''
train = pd.read_csv('your_path/train.csv', dtype = dtypes)
old_test = pd.read_csv('your_path/test_supplement.csv', dtype = dtypes)
train = pd.concat([train,old_test], axis=0, ignore_index=True)
del old_test
gc.collect()
#pre_processing
prepare(train)
train = get_user_info(train)
train = get_kernel_fe(train)
train = get_hourly_app_info(train)
train = get_click_time_info(train)
train = get_next_click_stat(train)
train = get_old_size(train)
train = get_old_unique(train)
train = get_yulia_fe(train)
train = train.sort_values(by=['row_id'])
gc.collect()
#split
train, old_test = train.iloc[:184903890,:], train.iloc[184903890:,:]
train.drop(['click_id'], axis=1, inplace=True)
old_test.drop(['attributed_time', 'is_attributed'], axis=1, inplace=True)
old_test.rename(columns={'click_id': 'old_click_id'}, inplace=True)
#get model and predict(so you could do things such as model.predict(your_test[predictors]))
model, predictors = get_p1(train, 2200, if_ip=False)
'''
#then use your own way to map from test_supplement to test and use the returned model and predictors to get the prediction, that's all 