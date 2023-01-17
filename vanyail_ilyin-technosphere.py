import pandas as pd

import numpy as np
df_sample_submission = pd.read_csv("../input/sample_submission.csv")

df_store = pd.read_csv("../input/store.csv")

df_test = pd.read_csv("../input/test.csv")

df_train = pd.read_csv("../input/train.csv")
s = df_train['Open'] != 0 

df_train = df_train[s]
df_train['Year'] = df_train['Date'].apply(lambda x: int(x[0:4]))

df_train['Month'] = df_train['Date'].apply(lambda x: int(x[5:7]))

print(df_train.groupby(['Year'])['Sales'].agg('mean'))

print(df_train.groupby(['Month'])['Sales'].agg('mean'))

print(df_train.groupby(['StateHoliday'], axis=0)['Sales'].agg('mean')) # Важный столбец. 

state_holiday = pd.get_dummies(df_train['StateHoliday'],prefix="state_holidays",prefix_sep='_')

print(df_train.groupby(['DayOfWeek'], axis=0)['Sales'].agg('mean')) # Тоже важны столбец - в вскр. нет продаж. 

day_of_week = pd.get_dummies(df_train['DayOfWeek'],prefix="day_of_week_") #

month = pd.get_dummies(df_train['Month'],prefix="month_")

del df_train['Date']

del df_train['DayOfWeek']

del df_train['Year'] #пока что без этого поля, цифры снизу показывают, что оно не особо важно

#А вот колебания продаж по месяцам довольно сильные

del df_train['StateHoliday']

df_train = pd.concat([df_train,day_of_week,state_holiday,month], axis=1)
print(df_train.groupby(['Month'], axis=0)['Sales'].mean()/df_train.groupby(['Month'], axis=0)['Customers'].mean())

del df_train['Month'] 
df_test['Year'] = df_test['Date'].apply(lambda x: int(x[0:4]))

df_test['Month'] = df_test['Date'].apply(lambda x: int(x[5:7]))

state_holiday = pd.get_dummies(df_test['StateHoliday'],prefix="state_holidays",prefix_sep="_")

day_of_week = pd.get_dummies(df_test['DayOfWeek'],prefix="day_of_week_") #

month = pd.get_dummies(df_test['Month'],prefix="month_")

del df_test['Date']

del df_test['DayOfWeek']

del df_test['Year'] #пока что без этого поля, цифры снизу показывают, что оно не особо важно

#А вот колебания продаж по месяцам довольно сильные

del df_test['StateHoliday']

del df_test['Month'] 

df_test = pd.concat([df_test,day_of_week,state_holiday,month], axis=1)
del df_train['state_holidays_0']

del df_test['state_holidays_0']
y_train = df_train['Sales'].values

del df_train['Sales']

del df_train['Customers']
print(list(df_test))

print(list(df_train))
diff = list(set(list(df_train)) - set(list(df_test)))

print(diff)
for column in diff:

    df_test[column]=0
storetype = pd.get_dummies(df_store['StoreType'],prefix="storetype_")

assortment=pd.get_dummies(df_store['Assortment'],prefix="assortment_")

del df_store['StoreType']

del df_store['Assortment']

df_store = pd.concat([df_store,storetype,assortment], axis=1)
del df_store['Promo2SinceWeek']

del df_store['Promo2SinceYear']

del df_store['PromoInterval']

del df_store['CompetitionOpenSinceYear']

del df_store['CompetitionOpenSinceMonth'] #Удаляем столбцы, где сплошные NaN-ы 
df_test = pd.merge(df_test, df_store, how='left', on=['Store'])

df_train = pd.merge(df_train, df_store, how='left', on=['Store'])
del df_test['Store'] 

del df_test['Id']

del df_train['Store'] 
df_train = df_train.reindex_axis(sorted(df_train.columns), axis=1)
df_test = df_test.reindex_axis(sorted(df_test.columns), axis=1)
X_train = df_train.values

X_test = df_test.values
for i in range(0,X_train.shape[1]):

    s = np.isnan(X_train[:,i])

    print(s.sum())
Expectation = np.nanmean(X_train[:,0])

nans = np.isnan(X_train[:,0])

X_train[nans,0] = Expectation

s = np.isnan(X_train[:,0])
for i in range(0,X_test.shape[1]):

    s = np.isnan(X_test[:,i])

    print(s.sum())
Expectation = np.nanmean(X_test[:,0])

nans = np.isnan(X_test[:,0])

X_test[nans,0] = Expectation

s = np.isnan(X_test[:,0])
s = X_test[:,1] != 0 
X_test.shape[0]-s.sum()
nans = np.isnan(X_test[:,1])

X_test[nans,1] = 0
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10, 111, 20), 'max_depth':range(10,21,5)}

rf = RandomForestRegressor()

model = GridSearchCV(rf,parameters)

model.fit(X_train[:20000,],y_train[:20000])
print(model.best_params_)

print(model.best_score_)
model = RandomForestRegressor(n_estimators=70,max_depth=20)

model.fit(X_train,y_train)
s = X_test[:,1] == 0 
y = model.predict(X_test)
y[s] = 0
np.savetxt("submit.csv", np.dstack((np.arange(1, y.size+1),y))[0],"%d,%d",header="Id,Sales")