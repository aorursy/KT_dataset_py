import pandas as pd

import sqlite3

import datetime, timeit
conn = sqlite3.connect('../input/steam-spy-data-from-api-request/KS-Steam-Connection-201801.sqlite')

Currencies = pd.read_sql('select * from Currencies',conn)

conn.close()
ksname = 'ks-projects-201801'

KSprojects = pd.read_csv('../input/kickstarter-projects/'+ksname+'.csv',header=0)
KSprojects.deadline = KSprojects.deadline.astype('datetime64[ns]')

Currencies.date = Currencies.date.astype('datetime64[ns]')



Currencies=Currencies[['date']+[x for x in KSprojects.currency.unique() if x!='USD']]

Currencies.set_index('date',inplace=True)
def get_currency(d,cur):

    currency_data = Currencies.truncate(before=d.date())

    # if the campaign was ongoing at the time of data collection, the deadline could be in the future,

    # so we don't have the exchange rate for it

    # in that case we will use the most recent datapoint

    try:

        return currency_data.iloc[0][cur]

    except:

        return Currencies.iloc[-1][cur]
start_time = timeit.default_timer()

KSprojects['usd_pledged_real'] = KSprojects.apply(lambda x: x.pledged if x.currency=='USD' else x.pledged/get_currency(x.deadline,x.currency),axis=1)

elapsed = timeit.default_timer() - start_time

print('USD pledge values generated in',elapsed,'seconds.')