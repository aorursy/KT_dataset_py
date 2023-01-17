import numpy as np 
import pandas as pd 
import datetime
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import gc
%matplotlib inline

#make wider graphs
sns.set(rc={'figure.figsize':(12,5)});
plt.figure(figsize=(12,5));
# eg:
#import some file
temp = pd.read_csv('../input/train_sample.csv')

#do something to the file
temp['os'] = temp['os'].astype('str')
#delete when no longer needed
del temp
#collect residual garbage
gc.collect()
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }

train = pd.read_csv('../input/train_sample.csv', dtype=dtypes)

#check datatypes:
train.info()
train = pd.read_csv('../input/train.csv', nrows=10000, dtype=dtypes)
train.head()
#plain skipping looses heading info.  It's OK for files that don't have headings, 
#or dataframes you'll be linking together, or where you make your own custom headings...
train = pd.read_csv('../input/train.csv', skiprows=5000000, nrows=1000000, header = None, dtype=dtypes)
train.head()
#but if you want to import the headings from the original file
#skip first 5mil rows, but use the first row for heading:
train = pd.read_csv('../input/train.csv', skiprows=range(1, 5000000), nrows=1000000, dtype=dtypes)
train.head()
import subprocess
#from https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python , Olafur's answer
def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

lines = file_len('../input/train.csv')
print('Number of lines in "train.csv" is:', lines)
#generate list of lines to skip
skiplines = np.random.choice(np.arange(1, lines), size=lines-1-1000000, replace=False)

#sort the list
skiplines=np.sort(skiplines)

#check our list
print('lines to skip:', len(skiplines))
print('remaining lines in sample:', lines-len(skiplines), '(remember that it includes the heading!)')

###################SANITY CHECK###################
#find lines that weren't skipped by checking difference between each consecutive line
#how many out of first 100000 will be imported into the csv?
diff = skiplines[1:100000]-skiplines[2:100001]
remain = sum(diff!=-1)
print('Ratio of lines from first 100000 lines:',  '{0:.5f}'.format(remain/100000) ) 
print('Ratio imported from all lines:', '{0:.5f}'.format((lines-len(skiplines))/lines) )
train = pd.read_csv('../input/train.csv', skiprows=skiplines, dtype=dtypes)
train.head()
del skiplines
gc.collect()
train['click_time'] = pd.to_datetime(train['click_time'])
train['attributed_time'] = pd.to_datetime(train['attributed_time'])
train.describe(include='all')
#round the time to nearest hour
train['click_rnd']=train['click_time'].dt.round('H')  

#check for hourly patterns
train[['click_rnd','is_attributed']].groupby(['click_rnd'], as_index=True).count().plot()
plt.title('HOURLY CLICK FREQUENCY');
plt.ylabel('Number of Clicks');

train[['click_rnd','is_attributed']].groupby(['click_rnd'], as_index=True).mean().plot()
plt.title('HOURLY CONVERSION RATIO');
plt.ylabel('Converted Ratio');
#set up an empty dataframe
df_converted = pd.DataFrame()

#we are going to work with chunks of size 1 million rows
chunksize = 10 ** 6

#in each chunk, filter for values that have 'is_attributed'==1, and merge these values into one dataframe
for chunk in pd.read_csv('../input/train.csv', chunksize=chunksize, dtype=dtypes):
    filtered = (chunk[(np.where(chunk['is_attributed']==1, True, False))])
    df_converted = pd.concat([df_converted, filtered], ignore_index=True, )

df_converted.info()
df_converted.head()
#wanted columns
columns = ['ip', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'is_attributed' : 'uint8',
        }

ips_df = pd.read_csv('../input/train.csv', usecols=columns, dtype=dtypes)
print(ips_df.info())
ips_df.head()
#processing part of the table is not a problem
ips_df[0:100][['ip', 'is_attributed']].groupby('ip', as_index=False).count()[:10]
size=100000
all_rows = len(ips_df)
num_parts = all_rows//size

#generate the first batch
ip_counts = ips_df[0:size][['ip', 'is_attributed']].groupby('ip', as_index=False).count()

#add remaining batches
for p in range(1,num_parts):
    start = p*size
    end = p*size + size
    if end < all_rows:
        group = ips_df[start:end][['ip', 'is_attributed']].groupby('ip', as_index=False).count()
    else:
        group = ips_df[start:][['ip', 'is_attributed']].groupby('ip', as_index=False).count()
    ip_counts = ip_counts.merge(group, on='ip', how='outer')
    ip_counts.columns = ['ip', 'count1','count2']
    ip_counts['counts'] = np.nansum((ip_counts['count1'], ip_counts['count2']), axis = 0)
    ip_counts.drop(columns=['count1', 'count2'], axis = 0, inplace=True)
#see what we've got:
ip_counts.head()
ip_counts.sort_values('counts', ascending=False)[:20]
np.sum(ip_counts['counts'])
size=100000
all_rows = len(ips_df)
num_parts = all_rows//size

#generate the first batch
ip_sums = ips_df[0:size][['ip', 'is_attributed']].groupby('ip', as_index=False).sum()

#add remaining batches
for p in range(1,num_parts):
    start = p*size
    end = p*size + size
    if end < all_rows:
        group = ips_df[start:end][['ip', 'is_attributed']].groupby('ip', as_index=False).sum()
    else:
        group = ips_df[start:][['ip', 'is_attributed']].groupby('ip', as_index=False).sum()
    ip_sums = ip_sums.merge(group, on='ip', how='outer')
    ip_sums.columns = ['ip', 'sum1','sum2']
    ip_sums['conversions_per_ip'] = np.nansum((ip_sums['sum1'], ip_sums['sum2']), axis = 0)
    ip_sums.drop(columns=['sum1', 'sum2'], axis = 0, inplace=True)
ip_sums.head(10)
#check proportion (we calculated earlier how many rows of data had conversions)
np.sum(ip_sums['conversions_per_ip'])/184900000
ip_conversions=ip_counts.merge(ip_sums, on='ip', how='left')
ip_conversions.head()
ip_conversions['converted_ratio']=ip_conversions['conversions_per_ip']/ip_conversions['counts']
ip_conversions[:10]
#some cleanup
del ip_conversions
del ip_sums
del ips_df
del df_converted
del train
gc.collect()
import dask
import dask.dataframe as dd
# Loading in the train data
dtypes = {'ip':'uint32',
          'app': 'uint16',
          'device': 'uint16',
          'os': 'uint16',
          'channel': 'uint16',
          'is_attributed': 'uint8'}

train = dd.read_csv("../input/train.csv", dtype=dtypes, parse_dates=['click_time', 'attributed_time'])
train.head()
train.info()
len(train)
train.columns
#select only rows 'is_attributed'==1
train[train['is_attributed']==1].head()
#select only data attributed after 2017-11-06 
train[train['attributed_time']>='2017-11-07 00:00:00'].head()
ip_counts = train.ip.value_counts().compute()
ip_counts[:20]
#clean up to free up space
#for future work, you can export data you generated to CSVs so you don't have to make it
#all over again
del ip_counts
gc.collect()
channel_means = train[['channel','is_attributed']].groupby('channel').mean().compute()
channel_means[:20]
channel_means=channel_means.reset_index()
channel_means[:20]