# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Loading Dataset
register_df = pd.read_table('../input/gaoxiao-b-dataset/user_register_log.txt',header=None, sep='\t',
                            names=['user_id','register_day','register_type','device_type'])
launch_df = pd.read_table('../input/gaoxiao-b-dataset/app_launch_log.txt', names=['user_id','launch_day'],header=None, sep='\t')
video_df = pd.read_table('../input/gaoxiao-b-dataset/video_create_log.txt', names=['user_id','create_day'],header=None, sep='\t')
activity_df = pd.read_table('../input/gaoxiao-b-dataset/user_activity_log.txt', header=None, sep='\t',
                            names=['user_id','activity_day','page','video_id','author_id','action_type'])
def find_continue_day(seq_df, prefix):
    continue_days_count_list = []
    continue_days_max_list = []
    continue_days_min_list = []
    continue_days_mean_list = []
    for idx, row in seq_df.iterrows():
        uid = row[0]
        seq = row[1]
        continue_days = []
        if(len(seq)==1):
            continue_days.append(1)
        else:
            start = 0
            for i in range(1,len(seq)):
                if((seq[i]!=seq[i-1]+1) or (i==len(seq)-1)):
                    continue_days.append(i-start)
                    start = i
        continue_days_count_list.append(len(continue_days))
        continue_days_max_list.append(max(continue_days))
        continue_days_min_list.append(min(continue_days))
        continue_days_mean_list.append(np.mean(continue_days))
    return pd.DataFrame({'user_id':seq_df.user_id, prefix+'_count_continue_day': continue_days_count_list,
                         prefix+'_max_continue_day': continue_days_max_list,  prefix+'_min_continue_day': continue_days_min_list,
                         prefix+'_mean_continue_day': continue_days_mean_list})
def find_interval_day(x_df, time, seq_df, prefix):
    interval_days_count_list = []
    interval_days_max_list = []
    interval_days_min_list = []
    interval_days_mean_list = []
    for idx, row in seq_df.iterrows():
        uid = row[0]
        seq = row[1]
        interval_days = []
        if(len(seq)==1):
            interval_days.append(time-int(x_df[x_df.user_id==uid].register_day))
        else:
            start = 0
            for i in range(1,len(seq)):
                if((seq[i]!=seq[i-1]+1) or (i==len(seq)-1)):
                    interval_days.append(seq[i]-seq[i-1])
                    start = i
        interval_days_count_list.append(len(interval_days))
        interval_days_max_list.append(max(interval_days))
        interval_days_min_list.append(min(interval_days))
        interval_days_mean_list.append(np.mean(interval_days))
    return pd.DataFrame({'user_id':seq_df.user_id, prefix+'_count_interval_day': interval_days_count_list,
                        prefix+'_max_interval_day': interval_days_max_list, prefix+'_min_interval_day': interval_days_min_list,
                        prefix+'_mean_interval_day': interval_days_mean_list})
def find_first_last_interval_day(x_df, time, seq_df, prefix):
    firstday_interval_list = []
    lastday_interval_list = []
    for idx, row in seq_df.iterrows():
        uid = row[0]
        seq = row[1]
        min_day = min(seq)
        max_day = max(seq)
        firstday_interval_list.append(min_day - int(x_df[x_df.user_id==uid].register_day))
        lastday_interval_list.append(time - max_day)
    return pd.DataFrame({'user_id': seq_df.user_id, prefix+'_firstday_interval':firstday_interval_list,
                        prefix+'_lastday_interval': lastday_interval_list})
def find_num_count_ratio(x_df, time, seq_df, prefix):
    num_list = []
    count_list = []
    ratio_list = []
    maxnum_oneday_list = []
    meannum_oneday_list = []
    for idx, row in seq_df.iterrows():
        uid = row[0]
        seq = row[1]
        num_list.append(len(seq))
        count_list.append(len(np.unique(seq)))
        ratio_list.append(len(np.unique(seq))/(time-int(x_df[x_df.user_id==uid].register_day)+1))
        maxnum_oneday_list.append(np.max(np.bincount(seq)))
        meannum_oneday_list.append(np.mean(np.bincount(seq)))
    return pd.DataFrame({'user_id':seq_df.user_id, prefix+'_num': num_list, prefix+'_count': count_list, prefix+'_ratio':ratio_list,
                        prefix+'_maxnum_oneday': maxnum_oneday_list, prefix+'meannum_oneday': meannum_oneday_list})
def GetLabel(register_df, launch_df, video_df, activity_df,time):
    launch_uid = launch_df[(launch_df.launch_day>=time) & (launch_df.launch_day<time+7)].user_id.unique()
    video_uid = video_df[(video_df.create_day>=time) & (video_df.create_day<time+7)].user_id.unique()
    activity_uid = activity_df[(activity_df.activity_day>=time) & (activity_df.activity_day>time+7)].user_id.unique()
    y = register_df.user_id.map(lambda x: x in launch_uid or x in activity_uid or x in video_uid ).map({False: 0, True:1})
    return pd.DataFrame(y)
def GetRegisterFeature(x_df, register_df):
    ### Register Type Feature ###
    register_type_dummies_df = pd.get_dummies(register_df['register_type'], prefix='register_type')
    ### Device Type Feature ###
    def device_type_map(x):
        if(device_type_count[x]>1500):
            return 0
        elif(device_type_count[x]>1000):
            return 1
        elif(device_type_count[x]>500):
            return 2
        else:
            return 3
    device_type_count = register_df['device_type'].value_counts()
    register_df['device_pop_type'] = register_df['device_type'].map(device_type_map)
    device_pop_type_dummies_df = pd.get_dummies(register_df['device_pop_type'], prefix='device_pop_type')
    ### merge 
    x_df = pd.concat([x_df, register_type_dummies_df, device_pop_type_dummies_df], axis=1)
    x_df = x_df.drop(['register_day'], axis=1)
    return x_df
def GetLaunchFeature(x_df, launch_df, time):

    launch_seq_unique_df = launch_df.groupby('user_id').apply(lambda x:np.sort(np.unique([int(n) for n in np.asarray(x)]))).reset_index()
    launch_seq_unique_df.columns = ['user_id', 'launch_seq']
    launch_seq_df = launch_df.groupby('user_id').apply(lambda x:np.sort([int(n) for n in np.asarray(x)])).reset_index()
    launch_seq_df.columns = ['user_id', 'launch_seq']
    
    ### video num,count,ratio,maxmean_oneday ###
    video_num_count_ratio_maxmean_oneday_df = find_num_count_ratio(x_df, time, launch_seq_df, prefix='launch')
    ### launch continue day ###
    launch_continue_day_df = find_continue_day(launch_seq_unique_df, prefix='launch')
    ### launch interval day ###
    launch_interval_day_df = find_interval_day(x_df, time, launch_seq_unique_df, prefix='launch')
    ### launch first last interval day ###
    launch_first_last_interval_day_df = find_first_last_interval_day(x_df, time, launch_seq_unique_df, prefix='launch')
    
    x_df = pd.merge(x_df, video_num_count_ratio_maxmean_oneday_df[['user_id','launch_count','launch_ratio']], on='user_id', how='left')
    x_df = pd.merge(x_df, launch_continue_day_df, on='user_id', how='left')
    x_df = pd.merge(x_df, launch_interval_day_df, on='user_id', how='left')
    x_df = pd.merge(x_df, launch_first_last_interval_day_df, on='user_id', how='left')
    
    ### fill na ###
    x_df = x_df.drop(['register_day'], axis=1)
    return x_df
def GetVideoFeature(x_df, video_df, time):
    video_uid = video_df.user_id.unique()
    video_bool_df = pd.concat([x_df.user_id, x_df.user_id.apply(lambda x: x in video_uid).map({False: 0, True:1})], axis=1)
    video_bool_df.columns = ['user_id', 'create_bool']
    
    video_seq_df = video_df.groupby('user_id').apply(lambda x:np.sort([int(n) for n in np.asarray(x)])).reset_index()
    video_seq_df.columns = ['user_id', 'video_seq']
    
    video_seq_unique_df = video_df.groupby('user_id').apply(lambda x:np.sort(np.unique([int(n) for n in np.asarray(x)]))).reset_index()
    video_seq_unique_df.columns = ['user_id', 'video_seq']
    
    ### video num,count,ratio,maxmean_oneday ###
    video_num_count_ratio_maxmean_oneday_df = find_num_count_ratio(x_df, time, video_seq_df, prefix='create')
    ### video continue day ###
    video_continue_day_df = find_continue_day(video_seq_unique_df, prefix='create')
    ### video interval day ###
    video_interval_day_df = find_interval_day(x_df, time, video_seq_unique_df, prefix='create')
    ### video first last interval day ###
    video_first_last_interval_day_df = find_first_last_interval_day(x_df, time, video_seq_unique_df, prefix='create')
    
    ### merge ###
    x_df = pd.merge(x_df, video_bool_df, on='user_id', how='left')
    x_df = pd.merge(x_df, video_num_count_ratio_maxmean_oneday_df, on='user_id', how='left')
    x_df = pd.merge(x_df, video_continue_day_df, on='user_id', how='left')
    x_df = pd.merge(x_df, video_interval_day_df, on='user_id', how='left')
    x_df = pd.merge(x_df, video_first_last_interval_day_df, on='user_id', how='left')
    
    ### fill na ###
    
    x_df.fillna({'create_firstday_interval':time, 'create_lastday_interval':time,
                'create_max_interval_day':time, 'create_min_interval_day':time,
                'create_mean_interval_day':time}, inplace=True)
    x_df.fillna(0, inplace=True)
    x_df = x_df.drop(['register_day'], axis=1)
    return x_df
def GetActivityFeature(x_df, activity_df, time):
    activity_uid = activity_df.user_id.unique()
    activity_bool_df = pd.concat([x_df.user_id, x_df.user_id.apply(lambda x: x in activity_uid).map({False: 0, True:1})], axis=1)
    activity_bool_df.columns = ['user_id', 'activity_bool']
    activity_seq_df = activity_df[['user_id','activity_day']].groupby('user_id').apply(lambda x:np.sort([int(n) for n in np.asarray(x)])).reset_index()
    activity_seq_df.columns = ['user_id', 'activity_seq']
    
    activity_seq_unique_df = activity_df[['user_id','activity_day']].groupby('user_id').apply(lambda x:np.sort(np.unique([int(n) for n in np.asarray(x)]))).reset_index()
    activity_seq_unique_df.columns = ['user_id', 'activity_seq']
    
    ### video num,count,ratio,maxmean_oneday ###
    activity_num_count_ratio_maxmean_oneday_df = find_num_count_ratio(x_df, time, activity_seq_df, prefix='activity')
    ### video continue day ###
    activity_continue_day_df = find_continue_day(activity_seq_unique_df, prefix='activity')
    ### video interval day ###
    activity_interval_day_df = find_interval_day(x_df, time, activity_seq_unique_df, prefix='activity')
    ### video first last interval day ###
    activity_first_last_interval_day_df = find_first_last_interval_day(x_df, time, activity_seq_unique_df, prefix='activity')
    
    def get_dummies(df, prefix):
        df['bool'] = 1
        new_df = df.groupby(['user_id', prefix]).sum().reset_index()
        dummies_df = pd.DataFrame(index=df.user_id.unique())
        x = (df.groupby(['user_id', prefix]).sum()/df.groupby(['user_id']).sum()).drop([prefix], axis=1).reset_index()
        for idx,row in x.iterrows():
            dummies_df.loc[int(row.user_id), prefix+'_'+str(int(row[prefix]))]=float(row['bool'])
        dummies_df = dummies_df[np.sort(np.asarray(dummies_df.columns))]
        dummies_df.reset_index(inplace=True)
        dummies_df.rename(columns={'index':'user_id'},inplace=True)
        return dummies_df

    ### 
    page_dummies_df = get_dummies(activity_df[['user_id', 'page']], 'page')
    action_type_dummies_df = get_dummies(activity_df[['user_id', 'action_type']], 'action_type')
    
    author_id_df_rmdup = activity_df[['user_id', 'author_id']].drop_duplicates()
    author_id_count_df = author_id_df_rmdup.groupby('user_id').count().reset_index().rename(columns={'author_id': 'author_id_count'})

    video_id_df_rmdup = activity_df[['user_id','video_id']].drop_duplicates()
    video_id_count_df = video_id_df_rmdup.groupby('user_id').count().reset_index().rename(columns={'video_id': 'video_id_count'})

    video_author_id_df = pd.merge(video_id_count_df,author_id_count_df, how='left',on='user_id')
    video_author_id_df['video_author_ratio'] = video_author_id_df['video_id_count']/video_author_id_df['author_id_count']


    
    ### merge ###
    x_df = pd.merge(x_df, activity_bool_df, on='user_id', how='left')
    x_df = pd.merge(x_df, activity_num_count_ratio_maxmean_oneday_df, on='user_id', how='left')
    x_df = pd.merge(x_df, activity_continue_day_df, on='user_id', how='left')
    x_df = pd.merge(x_df, activity_interval_day_df, on='user_id', how='left')
    x_df = pd.merge(x_df, activity_first_last_interval_day_df, on='user_id', how='left')
    x_df = pd.merge(x_df, page_dummies_df, on='user_id', how='left')
    x_df = pd.merge(x_df, action_type_dummies_df, on='user_id', how='left')
    x_df = pd.merge(x_df, video_author_id_df, on='user_id', how='left')
    
    ### fill na ###
    x_df.fillna({'activity_firstday_interval':time, 'activity_lastday_interval':time,
                 'activity_max_interval_day':time,'activity_min_interval_day':time,
                 'activity_mean_interval_day':time}, inplace=True)
    x_df.fillna(0, inplace=True)
    x_df = x_df.drop(['register_day'], axis=1)
    return x_df
# Dataset splition
def GetDataFeature(time, register_df, launch_df, video_df, activity_df, flag='train'):
    tr_time = time
    tr_register_df = register_df[register_df.register_day<tr_time]
    tr_launch_df = launch_df[(launch_df.launch_day>tr_time-17) & (launch_df.launch_day<tr_time)]
    tr_video_df = video_df[(video_df.create_day>tr_time-17) & (video_df.create_day<tr_time)]
    #print(tr_launch_df.shape)
    tr_activity_df = activity_df[(activity_df.activity_day>tr_time-17) & (activity_df.activity_day<tr_time)]
    tr_x_df = tr_register_df[['user_id','register_day']]
    tr_register_feature_df = GetRegisterFeature(tr_x_df, tr_register_df)
    tr_launch_feature_df = GetLaunchFeature(tr_x_df, tr_launch_df, tr_time)
    tr_video_feature_df = GetVideoFeature(tr_x_df, tr_video_df, tr_time)
    tr_activity_feature_df = GetActivityFeature(tr_x_df, tr_activity_df, tr_time)
    tr_x_df = pd.merge(tr_x_df[['user_id','register_day']], tr_register_feature_df, on='user_id', how='left')
    tr_x_df = pd.merge(tr_x_df, tr_launch_feature_df, on='user_id', how='left')
    tr_x_df = pd.merge(tr_x_df, tr_video_feature_df, on='user_id', how='left')
    tr_x_df = pd.merge(tr_x_df, tr_activity_feature_df, on='user_id', how='left')
    
    if(flag=='test'):
        return tr_x_df
    else:
        tr_y_df =  GetLabel(tr_register_df, launch_df, video_df, activity_df, tr_time)
        return tr_x_df, tr_y_df
tr_x, tr_y = GetDataFeature(17, register_df, launch_df, video_df, activity_df, flag='train')
tr_x.insert(list(tr_x.columns).index('register_type_11'),'register_type_10', 0)
tr_x.insert(list(tr_x.columns).index('device_pop_type_1'),'device_pop_type_0', 0)
va_x, va_y = GetDataFeature(24, register_df, launch_df, video_df, activity_df, flag='valid')
te_x = GetDataFeature(31, register_df, launch_df, video_df, activity_df, flag='test')
print(tr_x.shape, tr_y.shape, va_x.shape, va_y.shape, te_x.shape)
tr_x.to_csv('tr_x.b.csv', index=None)
tr_y.to_csv('tr_y.b.csv', index=None)
va_x.to_csv('va_x.b.csv', index=None)
va_y.to_csv('va_y.b.csv', index=None)
te_x.to_csv('te_x.b.csv', index=None)