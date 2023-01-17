# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
input = pd.read_csv('/kaggle/input/open-shopee-code-league-logistic/delivery_orders_march.csv')

sla = {'Metro Manila': [3,5,7,7],

       'Luzon': [5,5,7,7],

       'Visayas': [7,7,7,7],

       'Mindanao': [7,7,7,7]

        }

df_sla = pd.DataFrame(sla, columns = ['Metro Manila','Luzon','Visayas','Mindanao'], index=['Metro Manila','Luzon','Visayas','Mindanao'])

df_sla.head()
input['new_buyeraddress'] = input['buyeraddress'].apply(lambda x : 'Metro Manila' if 'metro manila' in x.lower() else 

                                                     ('Luzon' if 'luzon' in x.lower() else

                                                      ('Visayas' if 'visayas' in x.lower() else

                                                      ('Mindanao' if 'mindanao' in x.lower() else 'Others'

                                                      ))))



input['new_selleraddress'] = input['selleraddress'].apply(lambda x : 'Metro Manila' if 'metro manila' in x.lower() else 

                                                     ('Luzon' if 'luzon' in x.lower() else

                                                      ('Visayas' if 'visayas' in x.lower() else

                                                      ('Mindanao' if 'mindanao' in x.lower() else 'Others'

                                                      ))))



input = input.drop(['buyeraddress', 'selleraddress'], axis=1)
def getsla(seller,buyer):

    return df_sla[seller][buyer]



input['sla'] = input.apply(lambda x: getsla(x.new_selleraddress,x.new_buyeraddress),axis=1)
#Converting epoch to date

input['d_pick'] = pd.to_datetime(input['pick'],unit='s').dt.floor('d')

input['d_1st_deliver_attempt'] = pd.to_datetime(input['1st_deliver_attempt'],unit='s').dt.floor('d')

input['d_2nd_deliver_attempt'] = pd.to_datetime(input['2nd_deliver_attempt'],unit='s').dt.floor('d')

input.head()
input = input.drop(['pick', '1st_deliver_attempt','2nd_deliver_attempt'], axis=1)
def num_days_between( start, end, week_day):

    num_weeks, remainder = divmod( (end-start).days, 7)

    if ( week_day - start.weekday() ) % 7 <= remainder:

       return num_weeks + 1

    else:

       return num_weeks

from datetime import datetime, timedelta 

input['sundays_1'] = input.apply(lambda x: num_days_between(x.d_pick+timedelta(days=1),x.d_1st_deliver_attempt,6),axis=1)

input['sundays_2'] = input.apply(lambda x: num_days_between(x.d_1st_deliver_attempt+timedelta(days=1)  ,x.d_2nd_deliver_attempt,6),axis=1)
import datetime

def getph( start, end):

    return int(start<datetime.datetime(2020,3,25,0,0,0)<=end)+int(start<datetime.datetime(2020,3,30,0,0,0)<=end)+int(start<datetime.datetime(2020,3,31,0,0,0)<=end)
input['ph_1'] = input.apply(lambda x: getph(x.d_pick,x.d_1st_deliver_attempt),axis=1)
input['ph_2'] = input.apply(lambda x: getph(x.d_1st_deliver_attempt,x.d_2nd_deliver_attempt),axis=1)
input['time_1'] = (input['d_1st_deliver_attempt'] - input['d_pick']).dt.days - input['ph_1'] - input['sundays_1']
input['time_2'] = (input['d_2nd_deliver_attempt'] - input['d_1st_deliver_attempt']).dt.days - input['ph_2'] - input['sundays_2']
input = input.drop(['d_pick', 'd_1st_deliver_attempt','d_2nd_deliver_attempt','ph_1','ph_2','sundays_1','sundays_2','new_buyeraddress','new_selleraddress'], axis=1)
input.head()
not_late = input[(input['time_1'] <= input['sla']) & (input['time_2'] <= 3)][['orderid']]

not_late['is_late'] = 0

not_late.head()
late = input[~input['orderid'].isin(not_late['orderid'])][['orderid']]

late['is_late'] = 1

late.head()
#input['is_late1']=input.apply(lambda x: ((input['time_1']>input['sla'])),axis=1)

output = pd.concat([late, not_late], axis=0)
#input['is_late'] = input['is_late'].astype(int)

output.head()
output.to_csv('/kaggle/working/teamtwo.csv',index=False)