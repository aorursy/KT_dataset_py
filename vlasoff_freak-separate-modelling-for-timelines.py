import pandas as pd

import numpy as np





import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



from tqdm import tqdm_notebook 

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import graph_objs as go

init_notebook_mode(connected = True)
tra = pd.read_csv('../input/air_visit_data.csv')

tst = pd.read_csv('../input/sample_submission.csv')

hol = pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
tra['visit_date'] = pd.to_datetime(tra['visit_date'])

tra['dow'] = tra['visit_date'].dt.dayofweek

tra['year'] = tra['visit_date'].dt.year

tra['day'] = tra['visit_date'].dt.day

tra['month'] = tra['visit_date'].dt.month

tra['visit_date'] = tra['visit_date'].dt.date



tst['visit_date'] = tst['id'].map(lambda x: str(x).split('_')[2])

tst['air_store_id'] = tst['id'].map(lambda x: '_'.join(x.split('_')[:2]))

tst.drop('id', axis=1, inplace=True)

tst['visit_date'] = pd.to_datetime(tst['visit_date'])

tst['dow'] = tst['visit_date'].dt.dayofweek

tst['year'] = tst['visit_date'].dt.year

tst['day'] = tst['visit_date'].dt.day

tst['month'] = tst['visit_date'].dt.month

tst['visit_date'] = tst['visit_date'].dt.date

tst = tst[tra.columns]
hol['visit_date'] = pd.to_datetime(hol['visit_date'])

hol['day_of_week'] = hol['day_of_week']

hol['visit_date'] = hol['visit_date'].dt.date

train = pd.merge(tra, hol, how='left', on=['visit_date']) 

test = pd.merge(tst, hol, how='left', on=['visit_date'])
LE = LabelEncoder()

train['air_store_id'] = LE.fit_transform(train['air_store_id'])

test['air_store_id'] = LE.transform(test['air_store_id'])



train['day_of_week'] = LE.fit_transform(train['day_of_week'])

test['day_of_week'] = LE.transform(test['day_of_week'])
train.head(3)
test.head(3)
def plotly_df(df, title = 'Visitors'):

    data = []



    #for column in df.columns:

    trace = go.Scatter(

            x = df.visit_date,

            y = df.visitors,

            mode = 'lines',

            name = 'visitors')

    data.append(trace)



    layout = dict(title = title)

    fig = dict(data = data, layout = layout)

    iplot(fig, show_link=False)
time_series = train[train.air_store_id == train.air_store_id.unique()[17]][['visit_date', 'visitors']]

plotly_df(time_series, title = "Visitors")
time_series = train[train.air_store_id == train.air_store_id.unique()[84]][['visit_date', 'visitors']]

plotly_df(time_series, title = "Visitors")
print('Test dataset contains ', test.air_store_id.unique().shape[0], ' unique id and train ', 

      train.air_store_id.unique().shape[0])
id_from_test = test.air_store_id.unique()

id_from_train = train.air_store_id.unique()



for i in range(len(id_from_test)):

    sver = id_from_test[i]

    alert = 1

    for j in range(len(id_from_train)):

        if id_from_train[j] == sver:

            alert = 0

    if alert == 1:

        print('In train dataset absent restoraunt: ', test.air_store_id.unique()[i])

print('End!')
train = train.fillna(0)

test = test.fillna(0)
col = [c for c in train if c not in ['visit_date','visitors']]
X_for_submission = test[col]
model = GradientBoostingRegressor(loss='ls', n_estimators=200, random_state=1818)



metrics = []

sub = []





for i in tqdm_notebook(range(len(X_for_submission.air_store_id.unique()))):

    id_filter = X_for_submission.air_store_id.unique()[i]

    y_train = np.log(train[train.air_store_id == id_filter]['visitors']+1)

    X_train = train[train.air_store_id == id_filter][col]

    X_test = X_for_submission[X_for_submission.air_store_id == id_filter][col]





    if train[train.air_store_id == id_filter].shape[0] > 100:

        test_index = X_train.shape[0]

        X_con = pd.concat((X_train, X_test), ignore_index=True)



        y_test = pd.Series(np.zeros(X_con.shape[0]-test_index), name='Visitors')

        y_con = pd.concat((y_train, y_test), ignore_index=True)



        lag_start = 39

        lag_end   = 59

        

        

        for i in range(lag_start, lag_end):

                X_con["lag_{}".format(i)] = y_con.shift(i)

        

        X_train = X_con.loc[:test_index-1] 

        X_test = X_con.loc[test_index:]

        y_train = y_con.loc[:test_index-1]



        X_train['visitors'] = y_train

        X_train.dropna(inplace=True)

        col2 = [c for c in X_train if c not in ['visitors']]



        y_train = X_train['visitors']

        X_train = X_train[col2]



    

    

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=18, shuffle=False)

    model.fit(X_tr, y_tr)

    coef = mean_squared_error(y_val, model.predict(X_val))

    metrics = np.append(metrics, coef)

    pred = model.predict(X_test)

    

    sub = np.append(sub, pred)
print('Minimum of RMSLE = ', np.round(np.sqrt(metrics.min()), decimals=3), ', ', 

      'Mean of RMSLE = ', np.round(np.sqrt(metrics.mean()), decimals=3), ', ',

      'Max of RMSLE = ' ,np.round(np.sqrt(metrics.max()), decimals=3))
submission = pd.read_csv('../input/sample_submission.csv')

submission.visitors = np.exp(sub)-1

submission['visitors'] = submission['visitors'].apply(lambda x: 0 if x < 0 else x) 

submission.to_csv('submit.csv', index=False)

submission.head(7)