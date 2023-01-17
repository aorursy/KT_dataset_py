import pandas as pd #This is always assumed but is included here as an introduction.

import numpy as np

import matplotlib.pyplot as plt



lonja = pd.read_csv('../input/Lonja_vivo.csv', parse_dates=['datetime'])

lonja = lonja.set_index('datetime')

print('Dataset shape: {}'.format(lonja.shape))


lonja.plot(figsize=(10,5))

lonja.hist()
lonja['eur-kg'].isnull().value_counts()
lonja.describe()


cut = lonja.index[int(0.5*len(lonja))]

print('Mean before {}:'.format(cut))

print(lonja.loc[:cut].mean())

print('')

print('Mean after {}:'.format(cut))

print(lonja.loc[cut:].mean())

print('')

print('---------------------------')

print('')

print('Std before {}:'.format(cut))

print(lonja.loc[:cut].std())

print('')

print('Std after {}:'.format(cut))

print(lonja.loc[cut:].std())


from statsmodels.tsa.stattools import adfuller

result = adfuller(lonja['eur-kg'])

print('eur-kg')

print('--------------------------')

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))



print('\n\n')

from statsmodels.tsa.seasonal import seasonal_decompose as sd

sd_lonja = sd(lonja['eur-kg'], freq=52)





_=plt.figure(figsize=(15,10))

ax1=plt.subplot(311)

_=ax1.plot(sd_lonja.trend, label='eur-kg', alpha=0.7)

_=plt.legend()



ax2=plt.subplot(312)

_=ax2.plot(sd_lonja.seasonal, label='eur-kg', alpha=0.7)

_=plt.legend()



ax3=plt.subplot(313)

_=ax3.plot(sd_lonja.resid, label='eur-kg', alpha=0.7)

_=plt.legend()



sd_lonja.trend.describe()

sd_lonja.seasonal.describe()

sd_lonja.resid.describe()



lonja.plot(figsize=(10,5))
import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font='IPAGothic')

import numpy as np

import statsmodels.api as sm
tr_start = '2009-01-01'

tr_end = '2017-28-12'

te_start = '2018-04-01'

te_end = '2019-26-12'



lonja_train = lonja['2009-01-01':'2017-28-12']

lonja_test = lonja['2018-04-01':'2019-26-12']
lonja_test
lonja_train
# for this model, we'll try p=1, d=0 (stationary), q=1. P=2  (2 lags, that is, two years, of 52 weeks data points),

# D=1, Q=2 (2 years of Moving Average order), and m=52 (52 weeks per year)





#predictions

from statsmodels.tsa.arima_model import ARIMA

mod = sm.tsa.statespace.SARIMAX(lonja_train['eur-kg'], 

                                trend='n', 

                                order=(1,0,1), 

                                seasonal_order=(2,1,2,52), 

                                enforce_stationarity = False, 

                                enforce_invertibility = False)



results = mod.fit()

results.summary()



from sklearn.metrics import mean_squared_error
pred = results.predict(470,574)[1:]

print('ARIMAX model MSE:{}'.format(mean_squared_error(lonja_test,pred)))
pred.head
pred.plot()
lonja_test.plot()
lonja_test.size
lonja_test.size
pred.dtype
lonja_test.columns
prediccion = pd.DataFrame(pred)
prediccion[0]
test_y_pred = pd.DataFrame(lonja_test["eur-kg"])
test_y_pred.head()
test_y_pred["pred"] = pred.values
test_y_pred.head()
test_y_pred.plot()
# test_y_pred.to_csv('../input/predictions.csv')