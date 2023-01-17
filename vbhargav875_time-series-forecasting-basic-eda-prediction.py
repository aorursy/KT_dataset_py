import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Reshape, Dropout, LSTM, RepeatVector, TimeDistributed
train = pd.read_csv("../input/electricity-consumption/train.csv")
test = pd.read_csv("../input/electricity-consumption/test.csv")
train.head()
test.head()
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=train.electricity_consumption)

plt.figure(figsize=(10,4))
plt.xlim(train.temperature.min(), train.temperature.max()*1.1)
sns.boxplot(x=train.temperature)

plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=train.windspeed)

plt.figure(figsize=(10,4))
plt.xlim(train.var1.min(), train.var1.max()*1.1)
sns.boxplot(x=train.var1)

plt.figure(figsize=(10,4))
plt.xlim(train.pressure.min(), train.pressure.max()*1.1)
sns.boxplot(x=train.pressure)
train = train[train['electricity_consumption']<=1000]
train = train[train['var1']>=-30]
train = train[train['windspeed']<=400]
print(train.temperature.min())
print(train.temperature.max())
print(train.temperature.mean())
pyplot.plot(train.temperature)
print(train.pressure.min())
print(train.pressure.max())
print(train.pressure.mean())
pyplot.plot(train.pressure)
print(train.windspeed.min())
print(train.windspeed.max())
print(train.windspeed.mean())
pyplot.plot(train.windspeed)
print(train.var1.min())
print(train.var1.max())
print(train.var1.mean())
pyplot.plot(train.var1)
train.var2.unique()
le = LabelEncoder()
train["var2_label"] = le.fit_transform(train.var2)
train = train.drop('var2',axis=1)
le = LabelEncoder()
test["var2_label"] = le.fit_transform(test.var2)
test = test.drop('var2',axis=1)
train.head()
test.head()
train['DateAndTime'] = pd.to_datetime(train['datetime'])
test['DateAndTime'] = pd.to_datetime(test['datetime'])

train['month'] = train['DateAndTime'].dt.month
train['day'] = train['DateAndTime'].dt.day
train['year'] = train['DateAndTime'].dt.year
train['hour'] = train['DateAndTime'].dt.hour

test['month'] = test['DateAndTime'].dt.month
test['day'] = test['DateAndTime'].dt.day
test['year'] = test['DateAndTime'].dt.year
test['hour'] = test['DateAndTime'].dt.hour
train = train.drop('datetime',axis=1)
test = test.drop('datetime', axis=1)
train = train.drop('DateAndTime',axis=1)
test = test.drop('DateAndTime', axis=1)
train['temp_shift_1'] = train.groupby('day')['temperature'].shift(1)
train['pres_shift_1'] = train.groupby('day')['pressure'].shift(1)
train['ws_shift_1'] = train.groupby('day')['windspeed'].shift(1)
train['var1_shift_1'] = train.groupby('day')['var1'].shift(1)
train['var2_label_shift_1'] = train.groupby('day')['var2_label'].shift(1)
test['temp_shift_1'] = test.groupby('day')['temperature'].shift(1)
test['pres_shift_1'] = test.groupby('day')['pressure'].shift(1)
test['ws_shift_1'] = test.groupby('day')['windspeed'].shift(1)
test['var1_shift_1'] = test.groupby('day')['var1'].shift(1)
test['var2_label_shift_1'] = test.groupby('day')['var2_label'].shift(1)
train.isnull().sum().sum()
train = train.dropna()
train.isnull().sum().sum()
test.isnull().sum().sum()
test = test.fillna(test.mean())
test.isnull().sum().sum()
X = train.copy()
y = X['electricity_consumption']
X = X.drop(['electricity_consumption'],axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=18)
model = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.8,
              importance_type='split', learning_rate=0.025, max_depth=8,
              min_child_samples=20, min_child_weight=100, min_split_gain=0.0,
              n_estimators=1000, n_jobs=-1, num_leaves=256,
              random_state=None, reg_alpha=0.8, reg_lamda=0.8,silent=True)
model.fit(X_train, y_train, eval_metric='rmse')
# model = RandomForestRegressor(n_estimators=2000,random_state=42)
# model.fit(X_train, y_train)
#model = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05,objective='reg:squarederror',max_depth=10,reg_alpha=0,reg_lambda=1,booster='gbtree')
#model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.25,objective='reg:linear',max_depth=8,reg_alpha=0.7,reg_lambda=0.7,booster='gbtree')
#model = xgb.XGBRegressor(n_estimators=2000)
#model.fit(X_train, y_train,verbose=True, eval_set=[(X_train, y_train), (X_val, y_val)], early_stopping_rounds = 10)
#model.fit(X_train,y_train)
# _ = plot_importance(model, height=0.9)
# model = Sequential()
# model.add(Reshape((1,X_train.shape[1],1)))
# model.add(Conv2D(filters = 32, kernel_size = (1,1),padding = 'Same',
#              activation ='relu', input_shape = (1,X_train.shape[1],1)))
# model.add(Conv2D(filters = 32, kernel_size = (1,1),padding = 'Same',
#              activation ='relu'))
# model.add(Conv2D(filters = 32, kernel_size = (1,1),padding = 'Same',
#              activation ='relu'))
# model.add(Flatten())
# model.add(Dense (500, activation='relu'))
# model.add(Dense (10, activation='relu'))
# model.add(Dense (1, activation='linear'))
# model.compile(loss='mean_squared_error', optimizer='adam',
#               metrics=['mse'])
# CNNModel = model.fit(np.array(X_train), np.array(y_train), nb_epoch=10000, batch_size=20, validation_data=(np.array(X_val),np.array(y_val)))
predict_val = model.predict(X_val)
np.sqrt(mean_squared_error(y_val, predict_val))
predictions = model.predict(test)
submit = test.copy()
submit['electricity_consumption'] = predictions
submit = submit[['ID','electricity_consumption']]
submit.to_csv("submission.csv",index=False)