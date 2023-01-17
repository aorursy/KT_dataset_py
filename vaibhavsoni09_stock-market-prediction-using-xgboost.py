# Instaling nsepy - for share price data
# Installing ta - for technical indicators
#!pip install nsepy
#!pip install ta
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nsepy import get_history
from datetime import date
from nsepy.history import get_price_list,get_indices_price_list
import datetime as dt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,10
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV,KFold
from ta.momentum import RSIIndicator,StochasticOscillator
from fastai.tabular import add_datepart
import seaborn as sns
import altair as alt
sns.set()
%matplotlib inline

#remove warnings
import warnings
warnings.filterwarnings('ignore')
price = get_price_list(dt=date(2015,1,1))
# considering data from 2015 to 2016
data = get_history(symbol="TCS", start=date(2015,1,1), end=date(2016,12,31))
data.tail()
data_new=data.drop(['Symbol', 'Series','Last', 'VWAP', 'Turnover', 'Trades', 'Deliverable Volume',
       '%Deliverble'],axis=1)
def ChangeIndex(share):
  '''Changes Date index to datetime datatype'''
  share.index = pd.to_datetime(share.index) # chane index to datetime format
  pass
ChangeIndex(data_new)
data_new.reset_index(inplace=True)
data_new.head()
alt.Chart(data_new).mark_line(color='blue').encode(
    x='Date',
    y='Close'
  ).properties(
    height=400,
    width=1000
  )

add_datepart(data_new, 'Date', drop=False)
data_new.drop('Elapsed', axis=1, inplace=True) # not required for the model

def featurecalculator(share):
  share['EMA_9'] = share['Close'].ewm(9).mean() # exponential moving average of window 9
  share['SMA_5'] = share['Close'].rolling(5).mean() # moving average of window 5
  share['SMA_10'] = share['Close'].rolling(10).mean() # moving average of window 10
  share['SMA_15'] = share['Close'].rolling(15).mean() # moving average of window 15
  share['SMA_20'] = share['Close'].rolling(20).mean() # moving average of window 20
  share['SMA_25'] = share['Close'].rolling(25).mean() # moving average of window 25
  share['SMA_30'] = share['Close'].rolling(30).mean() # moving average of window 30
  EMA_12 = pd.Series(share['Close'].ewm(span=12, min_periods=12).mean())
  EMA_26 = pd.Series(share['Close'].ewm(span=26, min_periods=26).mean())
  share['MACD'] = pd.Series(EMA_12 - EMA_26)    # calculates Moving Average Convergence Divergence
  share['RSI'] = RSIIndicator(share['Close']).rsi() # calculates Relative Strength Index 
  share['Stochastic']=StochasticOscillator(share['High'],share['Low'],share['Close']).stoch() # Calculates Stochastic Oscillator
  pass
featurecalculator(data_new)
def labelencode(share):
  LE=LabelEncoder()
  share['Is_month_end']=LE.fit_transform(share['Is_month_end'])
  share['Is_month_start']=LE.fit_transform(share['Is_month_start'])
  share['Is_quarter_end']=LE.fit_transform(share['Is_quarter_end'])
  share['Is_quarter_start']=LE.fit_transform(share['Is_quarter_start'])
  share['Is_year_end']=LE.fit_transform(share['Is_year_end'])
  share['Is_year_start']=LE.fit_transform(share['Is_year_start'])
  pass
labelencode(data_new)
data_new.head(40)
# Dropping rows with Na values
data_new=data_new.iloc[33:]
data_new.reset_index(drop=True,inplace=True)
data_new.head()
data_new.drop(['Year','High','Low','Open','Prev Close','Volume','Date'],inplace=True,axis=1)
data_new.head()
data_new.columns
# Shifting the features a row up
data_new[['EMA_9', 'SMA_5', 'SMA_10', 'SMA_15',
       'SMA_20', 'SMA_25', 'SMA_30', 'MACD', 'RSI', 'Stochastic']]=data_new[['EMA_9', 'SMA_5', 'SMA_10', 'SMA_15',
       'SMA_20', 'SMA_25', 'SMA_30', 'MACD', 'RSI', 'Stochastic']].shift(-1)
data_new.head()
# Splitting the dataset into 70% training, 15% validation and 15% test
# train test split indexes
test_size  = 0.15
valid_size = 0.15

test_split_idx  = int(data_new.shape[0] * (1-test_size))
valid_split_idx = int(data_new.shape[0] * (1-(valid_size+test_size)))  


#train test split tcs

train= data_new.loc[:valid_split_idx]
valid= data_new.loc[valid_split_idx+1:test_split_idx]
test= data_new.loc[test_split_idx+1:]
y_train = train['Close']
X_train = train.drop(['Close'], 1)

y_valid = valid['Close']
X_valid = valid.drop(['Close'], 1)

y_test = test['Close']
X_test = test.drop(['Close'], 1)
parameters = {
    'n_estimators': [500,600],
    'learning_rate': [0.1],
    'max_depth': [8, 12, 15],
    'gamma': [ 0.005, 0.01,],
    'random_state': [42],
    'min_child_weight':[4,3],
    'subsample':[0.8,1],
    'colsample_bytree':[1],
    'colsample_bylevel':[1]
}
kfold=KFold(5)
eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = XGBRegressor(objective='reg:squarederror',n_jobs=-1)
clf = GridSearchCV(model, parameters,cv=kfold,scoring='neg_mean_absolute_error',verbose=0)

clf.fit(X_train, y_train)

print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')
model = XGBRegressor(**clf.best_params_, objective='reg:squarederror',n_jobs=-1)
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
y_pred=model.predict(X_test)
mean_absolute_error(y_test,y_pred)
params={'colsample_bylevel': 1,
 'colsample_bytree': 0.6,
 'gamma': 0.005,
 'learning_rate': 0.07,
 'max_depth': 10,
 'min_child_weight': 1,
 'n_estimators': 170,
 'random_state': 42,
 'subsample': 0.6}
eval_set = [(X_train, y_train), (X_valid, y_valid)]
xgb=XGBRegressor(**params, objective='reg:squarederror',n_jobs=-1)
xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)
y_pred = xgb.predict(X_test)
mean_absolute_error(y_test, y_pred)
plt.figure(figsize=(15,8))
sns.lineplot(y=y_pred,x=np.arange(69))
sns.lineplot(y=y_test,x=np.arange(69))
plt.legend(['Y-Predicted','Y-True'])
plt.title('Y-True vs Y-Predicted')
plt.show()
