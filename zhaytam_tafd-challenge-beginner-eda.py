# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.set(rc={'figure.figsize':(14,6)});
plt.figure(figsize=(14,6));

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Manually setting the types of columns reduces the memory usage by ~x2.7
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32' # for test data
}

# Read the training data as chunks of 4m
print('Reading the train.csv..')
reader = pd.read_csv('../input/train.csv', dtype=dtypes, chunksize=4000000,
                     usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'],
                     parse_dates=['click_time'])
chunks = [chunk for chunk in reader]

print('Selecting the train chunks to use..')
chunks_to_use = [chunks[x] for x in np.arange(0, len(chunks), 6)]
print('Selected {} chunks.'.format(len(chunks_to_use)))
train_df = pd.concat(chunks_to_use, ignore_index=True)
print('train_df created.')
print(train_df.info())

del reader, chunks, chunks_to_use
gc.collect()
print('~{:.2f}%'.format(len(train_df[train_df['is_attributed'] == 1]) * 100 / len(train_df)))
test_df = pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['click_time'])
data = train_df.append(test_df)
print('data created.')

train_df_len = len(train_df)
del train_df, test_df
gc.collect()
data.info()
data[['app', 'channel', 'click_time', 'device', 'ip', 'os']].isnull().sum()
unique_counts = data[['ip', 'app', 'channel', 'device', 'os']].apply(lambda x: x.unique().shape[0])
print(unique_counts)
plt.bar(unique_counts.index.values, unique_counts)

del unique_counts
gc.collect()
import pytz
cst = pytz.timezone('Asia/Shanghai')
data['local_click_time'] = data['click_time'].dt.tz_localize(pytz.utc).dt.tz_convert(cst)
data['click_day'] = data['local_click_time'].dt.day.astype('uint8')
data['click_hour'] = data['local_click_time'].dt.hour.astype('uint8')
data.drop(['click_time', 'local_click_time'], axis=1, inplace=True)
print('Extracted time features.')
clicks_per_ip = data['ip'].value_counts()[:20]
sns.barplot(clicks_per_ip.index.values, clicks_per_ip.values)
most_clicked_ips = clicks_per_ip[:2].index.values
fig, axes = plt.subplots(1, 2)

for i in range(len(most_clicked_ips)):
    temp_df = data[['ip', 'click_hour']][data['ip'] == most_clicked_ips[i]]
    sns.countplot(x='click_hour', data=temp_df, ax=axes[i])
    axes[i].set_title(most_clicked_ips[i])
ips_download_counts = data[['ip', 'app', 'is_attributed']][data['ip'].isin(most_clicked_ips)].groupby('ip').agg({ 'app': 'count', 'is_attributed': 'sum'})
ips_download_counts.rename(columns={'app': 'click_count', 'is_attributed': 'download_count'}, inplace=True)
ips_download_counts['download_rate'] = ips_download_counts['download_count'] * 100 / ips_download_counts['click_count']
ips_download_counts
data[['ip', 'device']][data['ip'].isin(most_clicked_ips)].groupby('ip')['device'].nunique()
temp_col = data[['ip', 'channel']].groupby('ip').count().reset_index().rename(columns={'channel': 'ip_count'}).astype('uint32')
data = data.merge(temp_col, on='ip', how='left')

del temp_col
gc.collect()
clicks_per_day_per_ip = data[['click_day', 'ip', 'channel']][data['ip'].isin(most_clicked_ips)].groupby(['click_day', 'ip']).count().rename(columns={'channel': 'count'})
clicks_per_day_per_ip.unstack().plot(kind='bar')
clicks_per_hour_per_ip = data[['click_hour', 'ip', 'channel']][data['ip'].isin(most_clicked_ips)].groupby(['click_hour', 'ip']).count().rename(columns={'channel': 'count'})
clicks_per_hour_per_ip.unstack().plot(kind='bar')
temp_col = data[['click_day', 'click_hour', 'ip', 'channel']].groupby(['click_day', 'click_hour', 'ip']).count().reset_index().rename(columns={'channel': 'day_hour_ip_count'}).astype('uint32')
data = data.merge(temp_col, on=['click_day', 'click_hour', 'ip'], how='left')

del temp_col
gc.collect()
del clicks_per_ip, ips_download_counts, clicks_per_day_per_ip, clicks_per_hour_per_ip
gc.collect()
clicks_per_device = data['device'].value_counts()[:10]
print(clicks_per_device)
sns.barplot(clicks_per_device.index.values, clicks_per_device.values)
most_used_devices = clicks_per_device.index.values
fig, axes = plt.subplots(2, 1)

for i in range(len(most_clicked_ips)):   
    temp_df = data[['ip', 'device']][data['ip'] == most_clicked_ips[i]]
    sns.countplot(x='device', data=temp_df, ax=axes[i], order=most_used_devices)
    axes[i].set_title(most_clicked_ips[i])
data[['ip', 'device', 'channel']].groupby(['ip', 'device'])
devices_download_counts = data[['device', 'ip', 'is_attributed']][data['device'].isin(most_used_devices)].groupby('device').agg({ 'ip': 'count', 'is_attributed': 'sum'})
devices_download_counts.rename(columns={'ip': 'click_count', 'is_attributed': 'download_count'}, inplace=True)
devices_download_counts['download_rate'] = devices_download_counts['download_count'] * 100 / devices_download_counts['click_count']
devices_download_counts
del clicks_per_device, devices_download_counts
gc.collect()
temp_col = data[['ip', 'device', 'channel']].groupby(['ip', 'device']).count().reset_index().rename(columns={'channel': 'ip_device_count'}).astype('uint32')
data = data.merge(temp_col, on=['ip', 'device'], how='left')

del temp_col
gc.collect()
clicks_per_app = data[['app', 'channel']].groupby('app').count().sort_values('channel', ascending=False)['channel']
print('Top 10', clicks_per_app[:10])
plt.scatter(clicks_per_app.index, clicks_per_app)
plt.xlabel('app')
plt.ylabel('count')
most_used_apps = clicks_per_app[:10].index.values
clicks_per_day_per_app = data[['click_day', 'app', 'ip']][data['app'].isin(most_used_apps)].groupby(['click_day', 'app']).count().rename(columns={'ip': 'count'})
clicks_per_day_per_app.unstack().plot(kind='bar')
clicks_per_hour_per_app = data[['click_hour', 'app', 'ip']][data['app'].isin(most_used_apps[:6])].groupby(['click_hour', 'app']).count().rename(columns={'ip': 'count'})
clicks_per_hour_per_app.unstack().plot(kind='bar')
apps_download_counts = data[['app', 'ip', 'is_attributed']][data['app'].isin(most_used_apps)].groupby('app').agg({ 'ip': 'count', 'is_attributed': 'sum'})
apps_download_counts.rename(columns={'ip': 'click_count', 'is_attributed': 'download_count'}, inplace=True)
apps_download_counts['download_rate'] = apps_download_counts['download_count'] * 100 / apps_download_counts['click_count']
apps_download_counts.sort_values('click_count', ascending=False)
temp_col = data[['click_day', 'click_hour', 'app', 'channel']].groupby(['click_day', 'click_hour', 'app']).count().reset_index().rename(columns={'channel': 'day_hour_app_count'}).astype('uint32')
data = data.merge(temp_col, on=['click_day', 'click_hour', 'app'], how='left')

del temp_col
gc.collect()
del clicks_per_app, clicks_per_day_per_app, clicks_per_hour_per_app, apps_download_counts
gc.collect()
clicks_per_channel = data[['app', 'channel']].groupby('channel').count().sort_values('app', ascending=False)['app']
print('Top 10', clicks_per_channel[:10])
plt.scatter(clicks_per_channel.index, clicks_per_channel)
plt.xlabel('channel')
plt.ylabel('count')
most_used_channels = clicks_per_channel[:10].index.values
clicks_per_app_per_channel = data[['app', 'channel', 'ip']][data['channel'].isin(most_used_channels[:6])][data['app'].isin(most_used_apps)].groupby(['app', 'channel']).count().rename(columns={'ip': 'count'})
clicks_per_app_per_channel.unstack().plot(kind='bar')
channel_download_counts = data[['channel', 'ip', 'is_attributed']][data['channel'].isin(most_used_channels)].groupby('channel').agg({ 'ip': 'count', 'is_attributed': 'sum'})
channel_download_counts.rename(columns={'ip': 'click_count', 'is_attributed': 'download_count'}, inplace=True)
channel_download_counts['download_rate'] = channel_download_counts['download_count'] * 100 / channel_download_counts['click_count']
channel_download_counts.sort_values('click_count', ascending=False)
temp_col = data[['app', 'channel', 'ip']].groupby(['app', 'channel']).count().reset_index().rename(columns={'ip': 'app_channel_count'}).astype('uint32')
data = data.merge(temp_col, on=['app', 'channel'], how='left')

del temp_col
gc.collect()
del clicks_per_channel, clicks_per_app_per_channel, channel_download_counts
gc.collect()
clicks_per_os = data[['os', 'channel']].groupby('os').count().sort_values('channel', ascending=False)['channel']
print('Top 10', clicks_per_os[:10])
plt.scatter(clicks_per_os.index, clicks_per_os)
plt.xlabel('os')
plt.ylabel('count')
most_used_os = clicks_per_os[:2].index.values
fig, axes = plt.subplots(2, 1)
for i in range(2):
    temp_df = data[['os', 'device']][data['os'] == most_used_os[i]]
    sns.countplot(x='device', data=temp_df, ax=axes[i], order=most_used_devices)
    axes[i].set_title(most_used_os[i])
temp_col = data[['os', 'device', 'channel']].groupby(['os', 'device']).count().reset_index().rename(columns={'channel': 'os_device_count'}).astype('uint32')
data = data.merge(temp_col, on=['os', 'device'], how='left')

del temp_col
gc.collect()
fig, axes = plt.subplots(2, 1)
for i in range(2):
    temp_df = data[['os', 'app']][data['os'] == most_used_os[i]][data['app'].isin(most_used_apps)]
    sns.countplot(x='app', data=temp_df, ax=axes[i], order=most_used_apps)
    axes[i].set_title(most_used_os[i])
temp_col = data[['os', 'app', 'channel', 'ip']].groupby(['os', 'app', 'channel']).count().reset_index().rename(columns={'ip': 'os_app_channel_count'}).astype('uint32')
data = data.merge(temp_col, on=['os', 'app', 'channel'], how='left')

del temp_col
gc.collect()
data[['ip', 'os']][data['os'].isin(most_used_os)].groupby('os').count()
fig, axes = plt.subplots(2, 1)
for i in range(2):
    temp_df = data[['os', 'ip']][data['ip'] == most_clicked_ips[i]]
    sns.countplot(x='os', data=temp_df, ax=axes[i], order=clicks_per_os[:10].index.values)
    axes[i].set_title(most_clicked_ips[i])
clicks_per_app_per_ip = data[['app', 'channel', 'ip']][data['ip'].isin(most_clicked_ips)][data['app'].isin(most_used_apps)].groupby(['app', 'ip']).count().rename(columns={'channel': 'count'})
clicks_per_app_per_ip = clicks_per_app_per_ip.reindex(most_used_apps, level='app')
clicks_per_app_per_ip.unstack().plot(kind='bar')