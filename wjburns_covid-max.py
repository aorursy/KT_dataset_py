# CORONA VIRUS - GROWTH RATE PREDICTION VIA TAYLOR SERIES MODEL

# REF: https://www.kaggle.com/rnglol/simple-taylor-series-model
%%time

# IMPORTS

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# LOAD TRAIN DATA

train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')



# SCRUB DATA

junk =['Id','Country/Region','Lat','Long','Province/State']

train.drop(junk, axis=1, inplace=True)
# PREP TRAIN DATA 

X_train = train[48:]

X_train.reset_index(inplace = True, drop = True) 

print(X_train)
# CALCULATE EXPANSION TABLE

diff_conf, conf_old = [], 0 

diff_fat, fat_old = [], 0

dd_conf, dc_old = [], 0

dd_fat, df_old = [], 0



for row in X_train.values:

    diff_conf.append(row[1]-conf_old)

    conf_old=row[1]

    diff_fat.append(row[2]-fat_old)

    fat_old=row[2]

    dd_conf.append(diff_conf[-1]-dc_old)

    dc_old=diff_conf[-1]

    dd_fat.append(diff_fat[-1]-df_old)

    df_old=diff_fat[-1]

    

print(len(diff_conf),X_train.shape)
# SAMPLES

samples = len(diff_conf)

answer = samples - 1

key = answer - 1
# POPULATE DATAFRAME FEATURES

pd.options.mode.chained_assignment = None  # default='warn'



X_train['diff_confirmed'] = diff_conf

X_train['diff_fatalities'] = diff_fat

X_train['dd_confirmed'] = dd_conf

X_train['dd_fatalities'] = dd_fat

    

X_train
# CALCULATE SERIES AVERAGES

d_c = X_train.diff_confirmed.drop(0).mean()

dd_c = X_train.dd_confirmed.drop(0).drop(1).mean()

d_f = X_train.diff_fatalities.drop(0).mean()

dd_f = X_train.dd_fatalities.drop(0).drop(1).mean()





print(d_c, dd_c, d_f, dd_f)
# ITERATE TAYLOR SERIES

pred_c, pred_f = list(X_train.ConfirmedCases.loc[2:answer]), list(X_train.Fatalities.loc[2:answer])



for i in range(1, 44 - key):

    pred_c.append(int((X_train.ConfirmedCases[answer] + (d_c + dd_c*i) * i) * 1.3))

    pred_f.append(int((X_train.Fatalities[answer] + (d_f + dd_f*i) * i)))
# WRITE SUBMISSION

my_submission = pd.DataFrame({'ForecastId': list(range(1,44)), 'ConfirmedCases': pred_c, 'Fatalities': pred_f})



my_submission.to_csv('submission.csv', index=False)
print(my_submission)