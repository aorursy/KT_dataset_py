import pandas as pd

import numpy as np

import sys

import re

df_key = pd.read_csv("../input/key.csv")

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df_weather = pd.read_csv("../input/weather.csv")



df_train['date'] = pd.to_datetime(df_train['date'])

df_weather['date'] = pd.to_datetime(df_weather['date'])



temp = pd.merge(df_train, df_key,how='left', on=['store_nbr'])

df_main_train = pd.merge(temp, df_weather, how='left', on=['station_nbr','date'])



print(df_train.shape)

print(temp.shape)

print(df_main_train.shape)

print(list(df_main_train))
df = df_main_train

df['year'], df['month'] = df['date'].dt.year, df['date'].dt.month

mask = (df['item_nbr'] == 6)

df = df.loc[mask]



df2 = df[['month','year','units']]



import matplotlib.pyplot as plt

count2 = df2.groupby(['month','year'])

totalsum = count2['units'].aggregate(np.sum).unstack()

##

x = totalsum.values.reshape(-1,1)

##

totalsum.plot(kind = 'bar', title = 'units')

plt.ylabel('count')

plt.show()
df3 = df[['preciptotal','month','year']]

df3['preciptotal'] = df3['preciptotal'].convert_objects(convert_numeric=True)

df3.interpolate()

import matplotlib.pyplot as plt



count3 = df3.groupby(['month','year'])

totalsum = count3['preciptotal'].aggregate(np.sum).unstack()

##

y = totalsum.values.reshape(-1,1)

##

totalsum.plot(kind = 'bar', title = 'preciptotal')

plt.ylabel('count')

plt.show()
#plt.plot(x, y, 'o', label="data")



x1 = df2['units'].values.reshape(-1,1)

y1 = df3['preciptotal'].values.reshape(-1,1)



plt.plot(x1, y1, 'o', label="data")
df7 = df_main_train



df7 = df7.convert_objects(convert_numeric=True)

df7.interpolate()





patternRA = 'RA'

patternSN = 'SN'

df7['RA'], df7['SN'] = df7['codesum'].str.contains(patternRA), df7['codesum'].str.contains(patternSN)

df7['Condition'] = (df7['RA'] & (df7['preciptotal']>1.0)) | (df7['SN'] & (df7['preciptotal']>2.0))



mask = (df7['Condition'] == True)

df8 = df7.loc[mask]



print(df8.shape)
df9 = df8[['date','preciptotal']]



df9.preciptotal.mean()



df10 = df9[np.abs(df9.preciptotal-df9.preciptotal.mean())>(3*df9.preciptotal.std())]



grouped_df = df10.groupby(['preciptotal'])['date']



for key, item in grouped_df:

    print(key)
df9 = df8[['date','tavg']]



df9['tavg'] = df9['tavg'].convert_objects(convert_numeric=True)

df9.interpolate()



df9.tavg.mean()



df10 = df9[np.abs(df9.tavg-df9.tavg.mean())>(3*df9.tavg.std())]



grouped_df = df10.groupby(['tavg'])['date']



for key, item in grouped_df:

    print(key)
df9 = df8[['date','avgspeed']]



df9['avgspeed'] = df9['avgspeed'].convert_objects(convert_numeric=True)

df9.interpolate()



df9.avgspeed.mean()



df10 = df9[np.abs(df9.avgspeed-df9.avgspeed.mean())>(3*df9.avgspeed.std())]



grouped_df = df10.groupby(['avgspeed'])['date']



for key, item in grouped_df:

    print(key)
import pandas as pd

import numpy as np

from patsy import dmatrices

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



df11 = df8



mask = (df11['item_nbr'] == 11)

df11 = df11.loc[mask]



df12 = df11[['units','tmax','tmin','tavg','depart','dewpoint','wetbulb','heat','cool']]

df12 = df12.convert_objects(convert_numeric=True).dropna()

df12 = df12._get_numeric_data()

df12.reset_index(drop=True)





df13 = df12[['tmax','tmin','tavg','depart','dewpoint','wetbulb','heat','cool']]



features = "+".join(df13.columns)

y, X = dmatrices('units ~' + features, df12, return_type='dataframe')



vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns



vif.round(1)


df11 = df8



mask = (df11['item_nbr'] == 11)

df11 = df11.loc[mask]



df12 = df11[['units','snowfall','preciptotal']]

df12 = df12.convert_objects(convert_numeric=True).dropna()

df12 = df12._get_numeric_data()

df12.reset_index(drop=True)



df13 = df12[['snowfall','preciptotal']]



features = "+".join(df13.columns)

y, X = dmatrices('units ~' + features, df12, return_type='dataframe')



vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns



vif.round(1)


df11 = df8



mask = (df11['item_nbr'] == 11)

df11 = df11.loc[mask]



df12 = df11[['units','stnpressure','sealevel','resultspeed','resultdir','avgspeed']]

df12 = df12.convert_objects(convert_numeric=True).dropna()

df12 = df12._get_numeric_data()

df12.reset_index(drop=True)



df13 = df12[['stnpressure','sealevel','resultspeed','resultdir','avgspeed']]



features = "+".join(df13.columns)

y, X = dmatrices('units ~' + features, df12, return_type='dataframe')



vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns



vif.round(1)