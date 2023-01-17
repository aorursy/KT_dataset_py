from fastai.tabular import *

import pandas as pd

import os, tarfile

import random

import matplotlib.pyplot as plt

import re

from datetime import *

from isoweek import Week



%matplotlib inline

%reload_ext autoreload

%autoreload 2



np.set_printoptions(threshold=50, edgeitems=20)

pd.options.mode.chained_assignment = None

PATH = '../input/multiple-data-source-of-rossmann/'

OUTPUT = './'
def rmax(n=50):

  pd.set_option('display.max_rows', n)



def cmax(n=50):

  pd.set_option('display.max_columns',n)
tarfile.open(f'{PATH}rossmann.tgz').extractall(path = OUTPUT)
table_names = ['train', 'store', 'store_states', 'state_names', 

               'googletrend', 'weather', 'test']

tables = [pd.read_csv(f'{OUTPUT}{fname}.csv', low_memory=False) for fname in table_names]

train, store, store_states, state_names, googletrend, weather, test = tables

train.shape, test.shape
from IPython.display import HTML, display

for t in tables: display(t.head())
train.StateHoliday.value_counts(), test.StateHoliday.value_counts()
train.StateHoliday = (train.StateHoliday!='0').astype('int')

test.StateHoliday = (test.StateHoliday!='0').astype('int')
googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
add_datepart(weather, "Date", drop=False)

add_datepart(googletrend, "Date", drop=False)

add_datepart(train, "Date", drop=False)

add_datepart(test, "Date", drop=False)
def join_df(left, right, left_on, right_on=None, suffix='_y'):

    if right_on is None: right_on = left_on

    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 

                      suffixes=("", suffix))
weather = join_df(weather, state_names, "file", "StateName")

weather.State.unique() , googletrend.file.unique()
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]

googletrend.loc[googletrend.State=='NI', "State"] = 'HB,NI'

train.shape, test.shape
trend_de = googletrend[googletrend.file == 'Rossmann_DE']
store = join_df(store, store_states, "Store")

len(store[store.State.isnull()])
joined = join_df(train, store, "Store")

joined_test = join_df(test, store, "Store")

len(joined[joined.StoreType.isnull()]),len(joined_test[joined_test.StoreType.isnull()])
joined = join_df(joined, googletrend, ["State","Year", "Week"])

joined_test = join_df(joined_test, googletrend, ["State","Year", "Week"])

len(joined[joined.trend.isnull()]),len(joined_test[joined_test.trend.isnull()])
joined = joined.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))

joined_test = joined_test.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))

len(joined[joined.trend_DE.isnull()]),len(joined_test[joined_test.trend_DE.isnull()])
joined = join_df(joined, weather, ["State","Date"])

joined_test = join_df(joined_test, weather, ["State","Date"])

len(joined[joined.Mean_TemperatureC.isnull()]),len(joined_test[joined_test.Mean_TemperatureC.isnull()])
joined.shape, joined_test.shape
for df in (joined, joined_test):

    for c in df.columns:

        if c.endswith('_y'):

            if c in df.columns: df.drop(c, inplace=True, axis=1)

        if c.endswith('_DE'):

            if not c.startswith('trend'):

              if c in df.columns: df.drop(c, inplace=True, axis=1)

joined.shape, joined_test.shape
for df in (joined,joined_test):

    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)

    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)

    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)

    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)
for df in (joined,joined_test):

    df["CompetitionOpenSince"] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear, 

                                                     month=df.CompetitionOpenSinceMonth, day=15))

    df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days

joined.CompetitionDaysOpen.describe()
joined.CompetitionDaysOpen.isna().any()
for df in (joined,joined_test):

    df.loc[df.CompetitionDaysOpen<0, "CompetitionDaysOpen"] = 0

    df.loc[df.CompetitionOpenSinceYear<1990, "CompetitionDaysOpen"] = 0

joined.CompetitionDaysOpen.describe() 
for df in (joined,joined_test):

    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"]//30

    df.loc[df.CompetitionMonthsOpen>24, "CompetitionMonthsOpen"] = 24

joined.CompetitionMonthsOpen.unique()

joined.CompetitionMonthsOpen.value_counts()
for df in (joined,joined_test):

    df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: Week(

        x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1).astype('datetime64'))

    df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days

for df in (joined,joined_test):

    df.loc[df.Promo2Days<0, "Promo2Days"] = 0

    df.loc[df.Promo2SinceYear<1990, "Promo2Days"] = 0

    df["Promo2Weeks"] = df["Promo2Days"]//7

    df.loc[df.Promo2Weeks<0, "Promo2Weeks"] = 0

    df.loc[df.Promo2Weeks>25, "Promo2Weeks"] = 25
joined.shape, joined_test.shape
columns = ["Date",'Dayofweek', "Store", "Promo", "StateHoliday", "SchoolHoliday"]

df = train[columns].append(test[columns])

df['StateHoliday'] = df.StateHoliday.astype('int')

df['Weekend'] = 0

df.loc[df.Dayofweek>= 5 ,['Weekend']] = 1
df['SchoolHoliday2'] = df['SchoolHoliday']

df['StateHoliday2'] = df['StateHoliday']

df['Promo2'] = df['Promo']



df.loc[df.Weekend==1, ['SchoolHoliday2', 'StateHoliday2', 'Promo2']] = 1
columns = ['SchoolHoliday2', 'StateHoliday2', 'Promo2']

sub = df[['Date','Store']+columns]   #make a smaller dataframe to work with. 

sub.sort_values(by=['Store', 'Date'], inplace=True)



daysum = sub.copy()



for c in columns:

  daysum.loc[:,c] = sub.groupby(['Store', sub[c].diff().ne(0).cumsum()])[c].transform('sum')



df = df.merge(daysum, how = 'left', on=['Date', 'Store'], suffixes=['', '_DaySum']) ;
rmax(500)

df[(df.Date>datetime(2015,3,1)) & (df.Store==1)].head(500)
rmax()
df.loc[df['SchoolHoliday2_DaySum'] > 2, 'SchoolHoliday'] = df['SchoolHoliday2']

df.loc[df['StateHoliday2_DaySum'] > 2, 'StateHoliday'] = df['StateHoliday2'] 

df.loc[df['Promo2_DaySum'] > 2, 'Promo'] = df['Promo2'] 
columns = ['SchoolHoliday', 'StateHoliday', 'Promo']

df = df[['Date','Store', 'Weekend']+columns]
def get_elapsed(fld, pre):

    day1 = np.timedelta64(1, 'D')

    last_date = np.datetime64()

    last_store = 0

    res = []



    for s,v,d in zip(df.Store.values,df[fld].values, df.Date.values):

        if s != last_store:

            last_date = np.datetime64()

            last_store = s

        if v: last_date = d

        res.append(((d-last_date).astype('timedelta64[D]') / day1))

    df[pre+fld] = res
fld = 'SchoolHoliday'

df = df.sort_values(['Store', 'Date'])

get_elapsed(fld, 'After')

df = df.sort_values(['Store', 'Date'], ascending=[True, False])

get_elapsed(fld, 'Before')

fld = 'StateHoliday'

df = df.sort_values(['Store', 'Date'])

get_elapsed(fld, 'After')

df = df.sort_values(['Store', 'Date'], ascending=[True, False])

get_elapsed(fld, 'Before')

fld = 'Promo'

df = df.sort_values(['Store', 'Date'])

get_elapsed(fld, 'After')

df = df.sort_values(['Store', 'Date'], ascending=[True, False])

get_elapsed(fld, 'Before')
df = df.set_index("Date")

columns = ['SchoolHoliday', 'StateHoliday', 'Promo']

for o in ['Before', 'After']:

    for p in columns:

        a = o+p

        df[a] = df[a].fillna(0).astype(int)
df.info()
columns = ['SchoolHoliday', 'StateHoliday', 'Promo']

sub = df[['Store']+columns]
# sub.sort_index(inplace=True)

sub.sort_values(by=['Store', 'Date'], inplace=True)

daysum = sub.copy()

daycount = sub.copy()



for c in columns:

  daysum.loc[:,c] = sub.groupby(['Store', sub[c].diff().ne(0).cumsum()])[c].transform('sum')

  daycount[c] = sub.groupby(['Store', sub[c].diff().ne(0).cumsum()])[c].transform('cumsum')



sub2 = sub.merge(daysum, how = 'left', on=['Date', 'Store'], suffixes=['', '_DaySum']) ;

sub2 = sub2.merge(daycount, how = 'left', on=['Date', 'Store'], suffixes=['', '_DayCount']) ; 
bwd = df[['Store']+columns].sort_index().groupby("Store").rolling(7, min_periods=1).sum()

fwd = df[['Store']+columns].sort_index(ascending=False

                                      ).groupby("Store").rolling(7, min_periods=1).sum()



bwd.drop(columns='Store',inplace=True)

bwd.reset_index(inplace=True)

fwd.drop(columns='Store',inplace=True)

fwd.reset_index(inplace=True)

df.reset_index(inplace=True)
df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])

df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])



sub2.reset_index(inplace=True)

sub2.drop(columns, 1 ,inplace=True)

df = df.merge(sub2, how='left', on=['Date','Store'])

df.drop(columns,1,inplace=True)
df.columns
df.info()
df.to_feather(f'{OUTPUT}df')
joined = join_df(joined, df, ['Store', 'Date'])

joined_test = join_df(joined_test, df, ['Store', 'Date'])
joined = joined[joined.Sales!=0]
joined.reset_index(drop=True, inplace=True)

joined_test.reset_index(drop=True, inplace=True)

joined.to_feather(f'{OUTPUT}joined2')

joined_test.to_feather(f'{OUTPUT}joined2_test')