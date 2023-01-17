import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# LOAD TRAIN DATA

train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

junk =['Id','Country/Region','Lat','Long','Province/State']

train.drop(junk, axis=1, inplace=True)
# LOAD MISSING CASE DATA

org = pd.read_csv('/kaggle/input/gthubdata-new/time_series_19-covid-Confirmed.csv')

us = org[org['Country/Region']=='US']

days = us.columns[4:]

ca_state = us[us['Province/State']=='California']

ca_counties = us[us['Province/State'].str.find('CA')>0]

missing=(ca_counties[days].sum() + ca_state[days])[days[4:48]]
missing
# ADD MISSING DATA

train.ConfirmedCases.loc[4]=int(missing['1/26/20'])

train.ConfirmedCases.loc[5]=int(missing['1/27/20'])

train.ConfirmedCases.loc[6]=int(missing['1/28/20'])

train.ConfirmedCases.loc[7]=int(missing['1/29/20'])

train.ConfirmedCases.loc[8]=int(missing['1/30/20'])

train.ConfirmedCases.loc[9]=int(missing['1/31/20'])

train.ConfirmedCases.loc[10]=int(missing['2/1/20'])

train.ConfirmedCases.loc[11]=int(missing['2/2/20'])

train.ConfirmedCases.loc[12]=int(missing['2/3/20'])

train.ConfirmedCases.loc[13]=int(missing['2/4/20'])

train.ConfirmedCases.loc[14]=int(missing['2/5/20'])

train.ConfirmedCases.loc[15]=int(missing['2/6/20'])

train.ConfirmedCases.loc[16]=int(missing['2/7/20'])

train.ConfirmedCases.loc[17]=int(missing['2/8/20'])

train.ConfirmedCases.loc[18]=int(missing['2/9/20'])

train.ConfirmedCases.loc[19]=int(missing['2/10/20'])

train.ConfirmedCases.loc[20]=int(missing['2/11/20'])

train.ConfirmedCases.loc[21]=int(missing['2/12/20'])

train.ConfirmedCases.loc[22]=int(missing['2/13/20'])

train.ConfirmedCases.loc[23]=int(missing['2/14/20'])

train.ConfirmedCases.loc[24]=int(missing['2/15/20'])

train.ConfirmedCases.loc[25]=int(missing['2/16/20'])

train.ConfirmedCases.loc[26]=int(missing['2/17/20'])

train.ConfirmedCases.loc[27]=int(missing['2/18/20'])

train.ConfirmedCases.loc[28]=int(missing['2/19/20'])

train.ConfirmedCases.loc[29]=int(missing['2/20/20'])

train.ConfirmedCases.loc[30]=int(missing['2/21/20'])

train.ConfirmedCases.loc[31]=int(missing['2/22/20'])

train.ConfirmedCases.loc[32]=int(missing['2/23/20'])

train.ConfirmedCases.loc[33]=int(missing['2/24/20'])

train.ConfirmedCases.loc[34]=int(missing['2/25/20'])

train.ConfirmedCases.loc[35]=int(missing['2/26/20'])

train.ConfirmedCases.loc[36]=int(missing['2/27/20'])

train.ConfirmedCases.loc[37]=int(missing['2/28/20'])

train.ConfirmedCases.loc[38]=int(missing['2/29/20'])

train.ConfirmedCases.loc[39]=int(missing['3/1/20'])

train.ConfirmedCases.loc[40]=int(missing['3/2/20'])

train.ConfirmedCases.loc[41]=int(missing['3/3/20'])

train.ConfirmedCases.loc[42]=int(missing['3/4/20'])

train.ConfirmedCases.loc[43]=int(missing['3/5/20'])

train.ConfirmedCases.loc[44]=int(missing['3/6/20'])

train.ConfirmedCases.loc[45]=int(missing['3/7/20'])

train.ConfirmedCases.loc[46]=int(missing['3/8/20'])

train.ConfirmedCases.loc[47]=int(missing['3/9/20'])
# LOAD MISSING DEATH DATA

xx = pd.read_csv('/kaggle/input/gthubdata-new/time_series_19-covid-Deaths.csv')

us_death = xx[xx['Country/Region']=='US']

days = us_death.columns[4:]

ca_state_death = us_death[us_death['Province/State']=='California']

ca_counties_death = us_death[us_death['Province/State'].str.find('CA')>0]

missing_death=(ca_counties_death[days].sum() + ca_state_death[days])[days[40:48]]
missing_death
train.Fatalities.loc[42]=int(missing_death['3/4/20'])

train.Fatalities.loc[43]=int(missing_death['3/5/20'])

train.Fatalities.loc[44]=int(missing_death['3/6/20'])

train.Fatalities.loc[45]=int(missing_death['3/7/20'])

train.Fatalities.loc[46]=int(missing_death['3/8/20'])

train.Fatalities.loc[47]=int(missing_death['3/9/20'])
train
# CALCULATE EXPANSION TABLE

diff_conf, conf_old = [], 0 

diff_fat, fat_old = [], 0

dd_conf, dc_old = [], 0

dd_fat, df_old = [], 0



for row in train.values:

    diff_conf.append(row[1]-conf_old)

    conf_old=row[1]

    diff_fat.append(row[2]-fat_old)

    fat_old=row[2]

    dd_conf.append(diff_conf[-1]-dc_old)

    dc_old=diff_conf[-1]

    dd_fat.append(diff_fat[-1]-df_old)

    df_old=diff_fat[-1]

    

print(len(diff_conf),train.shape)
# POPULATE DATAFRAME FEATURES

pd.options.mode.chained_assignment = None  # default='warn'



train['diff_confirmed'] = diff_conf

train['diff_fatalities'] = diff_fat

train['dd_confirmed'] = dd_conf

train['dd_fatalities'] = dd_fat

   

train = train[4:]

train.reset_index(inplace = True, drop = True) 



train
# CALCULATE SERIES MEAN AVERAGES

d_c = train.diff_confirmed.drop(0).mean()

dd_c = train.dd_confirmed.drop(0).drop(1).mean()

d_f = train.diff_fatalities.drop(0).mean()

dd_f = train.dd_fatalities.drop(0).drop(1).mean()



print("Daily New Cases:",d_c)

print("Case Variation:", dd_c)

print("Daily New Deaths:", d_f)

print("Death Variation:",dd_f)
# CURRENT SAMPLES

samples = len(train) #57

answer = samples - 1  #56

key_offset = answer - 45 #10
# ITERATE TAYLOR SERIES

pred_c, pred_f = list(train.ConfirmedCases.loc[46:answer]), list(train.Fatalities.loc[46:answer])



for i in range(1, 44 - key_offset):

    pred_c.append(int((train.ConfirmedCases[answer] + (d_c + dd_c*i) * i) * 1.3))

    pred_f.append(int((train.Fatalities[answer] + (d_f + dd_f*i) * i)))
# WRITE SUBMISSION

my_submission = pd.DataFrame({'ForecastId': list(range(1,44)), 'ConfirmedCases': pred_c, 'Fatalities': pred_f})

print(my_submission)

my_submission.to_csv('submission.csv', index=False)