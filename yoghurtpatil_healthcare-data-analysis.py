#Import the required libraries

import os

import random

import sqlite3

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#Connect to database and extract the exchange_transactions table 

os.chdir("/kaggle/input/healthcare-or-transactions-data")

sqlite_file = 'Healthcare OR Transactions Data.db'

conn = sqlite3.connect(sqlite_file)

extr = pd.read_sql_query("SELECT * FROM exchange_transactions", conn)
print("Dimensions of the exchange transactions table are: {}".format(extr.shape))
extr.describe()
#Drop duplicate rows

extr.drop_duplicates(keep='first', inplace=True)
#Convert relevant columns to datetime type and add a column for duration of each transaction

extr['created_datetime'] = pd.to_datetime(extr['created_datetime'])

extr['snapshot_date'] = pd.to_datetime(extr['snapshot_date'])

extr['start_time'] = pd.to_datetime(extr['start_time'])

extr['end_time'] = pd.to_datetime(extr['end_time'])

extr['duration'] = (extr['end_time'] - extr['start_time']).dt.total_seconds()/3600
#Calculate number of days of operation of iQueue

max(extr['snapshot_date']) - min(extr['snapshot_date'])
extr.shape[0] #Check the number of rows in the data after removing duplicates
extr.head() #Look at the top few rows of data
#Filter the transactions that do not have a parent transaction

fresh_tr = extr[extr['parent_transaction_id'].isnull()]

fresh_tr['action'].value_counts()
fresh_tr['location'].value_counts() #Count of locations
fresh_tr['room_name'].value_counts() #Count of rooms (top and bottom few)

requests_data = fresh_tr[fresh_tr['action']=='REQUEST']

req_dur = requests_data['duration']

release_data = fresh_tr[fresh_tr['action']=='RELEASE']

rel_dur = release_data['duration']



bins = np.linspace(0, 12, 30)

fig, ax = plt.subplots(1, 2)

ax[0].hist(req_dur, bins, alpha=0.5, color = 'black')

ax[0].set_ylabel('Count')

ax[0].set_xlabel('Request duration')

ax[0].set_ylim(0, 250)

ax[1].hist(rel_dur, bins, alpha=0.5, color = 'blue')

ax[1].set_ylim(0, 250)

ax[1].set_xlabel('Release duration')

plt.show()
transfer_data = fresh_tr[fresh_tr['action']=='TRANSFER']

plt.hist(transfer_data['duration'], color = 'green')

plt.xlabel('Transfer duration')

plt.ylabel('Count')

plt.show()
notnullpar_tr = extr[extr['parent_transaction_id'].notnull()]

notnullpar_tr['action'].value_counts()
fresh = fresh_tr[['transaction_id', 'created_datetime', 'action', 'duration']]

notnull = notnullpar_tr[['parent_transaction_id', 'created_datetime', 'action']]

fresh_leftjoin_notnull = fresh.merge(notnull, left_on='transaction_id', right_on='parent_transaction_id', how='left')

fresh_leftjoin_notnull.sort_values('transaction_id').head()
#Filter transactions having Transfer and Mark updated actions

transfer_upd = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='TRANSFER') & (fresh_leftjoin_notnull['action_y']=='MARK_UPDATED')]

transfer_upd.shape[0]
#Filter transactions having Transfer and Approve transfer actions

transfer_app = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='TRANSFER') & (fresh_leftjoin_notnull['action_y']=='APPROVE_TRANSFER')]

transfer_app.shape[0]
 #Filter transactions having Transfer and Deny transfer actions

transfer_deny = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='TRANSFER') & (fresh_leftjoin_notnull['action_y']=='DENY_TRANSFER')]

transfer_deny.shape[0]
transfer_app['duration'].sum()
transfer_upd.head()
transfer_updated = transfer_upd.copy()

transfer_updated.loc[:,'process_time'] = pd.Series((pd.to_datetime(transfer_updated['created_datetime_y']) - pd.to_datetime(transfer_updated['created_datetime_x'])).dt.total_seconds()/3600)

plt.hist(transfer_updated['process_time'])

plt.xlabel('Transfer processing time')

plt.ylabel('Frequency')

plt.show()
transfer_updated['process_time'].mean()
#Filter transactions having Release and Mark updated actions

release_upd = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='RELEASE') & (fresh_leftjoin_notnull['action_y']=='MARK_UPDATED')]

release_upd.shape[0]
#Filter transactions having Release and Deny release actions

release_den = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='RELEASE') & (fresh_leftjoin_notnull['action_y']=='DENY_RELEASE')]

release_den.shape[0]
release_upd['duration'].sum()
release_updated = release_upd.copy()

release_updated.loc[:,'process_time'] = pd.Series((pd.to_datetime(release_updated['created_datetime_y']) - pd.to_datetime(release_updated['created_datetime_x'])).dt.total_seconds()/3600)

plt.hist(release_updated['process_time'])

plt.xlabel('Release processing time')

plt.ylabel('Frequency')

plt.show()
release_updated['process_time'].mean()
#Filter transactions having Request and Mark updated actions

req_upd = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='REQUEST') & (fresh_leftjoin_notnull['action_y']=='MARK_UPDATED')]

req_upd.shape[0]
#Filter transactions having Request and Approve transfer actions

request_app = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='REQUEST') & (fresh_leftjoin_notnull['action_y']=='APPROVE_REQUEST')]

request_app.shape[0]
#Filter transactions having Request and Deny transfer actions

request_deny = fresh_leftjoin_notnull[(fresh_leftjoin_notnull['action_x']=='REQUEST') & (fresh_leftjoin_notnull['action_y']=='DENY_REQUEST')]

request_deny.shape[0]
request_app['duration'].sum()
request_approved = request_app.copy()

request_approved.loc[:,'process_time'] = pd.Series((pd.to_datetime(request_approved['created_datetime_y']) - pd.to_datetime(request_approved['created_datetime_x'])).dt.total_seconds()/3600)

plt.hist(request_approved['process_time'])

plt.xlabel('Request processing time')

plt.ylabel('Frequency')

plt.show()
request_approved['process_time'].mean()