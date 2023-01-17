import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc, time
import os

print(os.listdir("../input"))
dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}
def handleClickHour(df):
    df['click_hour']= (pd.to_datetime(df['click_time']).dt.round('H')).dt.hour
    df['click_hour'] = df['click_hour'].astype('uint16')
    df = df.drop(['click_time'], axis=1)   
    return df
def prepare_data():
    train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']

    #load training df (partly)
    start_time = time.time()
    df_train_30m = pd.read_csv('../input/train.csv', dtype=dtypes, skiprows=range(1,133333333), nrows=33333333, usecols=train_columns)
    print('Load df_train_30m with [{}] seconds'.format(time.time() - start_time))

    # Load testing df
    start_time = time.time()
    df_test = pd.read_csv('../input/test.csv', dtype=dtypes)
    print('Load df_test with [{}] seconds'.format(time.time() - start_time))

    train_record_index = df_train_30m.shape[0]

    #handle click hour 
    df_train_30m = handleClickHour(df_train_30m)
    df_test = handleClickHour(df_test)
    gc.collect()

    #df for submit
    df_submit = pd.DataFrame()
    df_submit['click_id'] = df_test['click_id']

    Learning_Y = df_train_30m['is_attributed']

    #drop zone
    df_test = df_test.drop(['click_id'], axis=1)
    df_train_30m = df_train_30m.drop(['is_attributed'], axis=1)
    gc.collect()

    df_merge = pd.concat([df_train_30m, df_test])
    del df_train_30m, df_test
    gc.collect()

    # Count ip for both train and test df 
    start_time = time.time()
    df_ip_count = df_merge['ip'].value_counts().reset_index(name='ip_count')
    df_ip_count.columns = ['ip', 'ip_count']
    print('Load df_ip_count with [{}] seconds'.format(time.time() - start_time))

    df_merge = df_merge.merge(df_ip_count, on='ip', how='left', sort=False)
    df_merge['ip_count'] = df_merge['ip_count'].astype('uint16')

    df_merge = df_merge.drop(['ip'], axis=1)
    del df_ip_count
    gc.collect()

    df_train = df_merge[:train_record_index]
    df_test = df_merge[train_record_index:]

    del df_merge
    gc.collect()
def train_forest():
    #Use RandomForest
    from sklearn.ensemble import RandomForestClassifier

    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=13, max_depth=13, random_state=13,verbose=2)
    rf.fit(df_train, Learning_Y)
    pickle.dump(rf, open( "rf.p", "wb"))
    print('Train RandomForest df_train_30m with [{}] seconds'.format(time.time() - start_time))

    #predict
    start_time = time.time()
    predictions = rf.predict_proba(df_test)
    print('Predict RandomForest df_train_22m with [{}] seconds'.format(time.time() - start_time))

    df_submit['is_attributed'] = predictions[:,1]
    df_submit.describe()

    df_submit.to_csv('random_forest_talking_data.csv', index=False)
import pickle

def load_forest():
    return pickle.load(open("rf.p", "rb"))    
import os.path

if os.path.isfile("rf.p"):
    rf = load_forest()
else:
    prepare_data()
    rf = train_forest()
