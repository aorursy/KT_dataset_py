# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
# Any results you write to the current directory are saved as output.
news_train_df.head()
print(len(news_train_df))
novel_news = news_train_df.loc[news_train_df.noveltyCount12H == 1,:]
print(len(novel_news))
novel_news = novel_news.loc[novel_news.sentimentClass == 1,:]
print(len(novel_news))
close = market_train_df.pivot(index='time', columns='assetCode', values='close')
open_ = market_train_df.pivot(index='time', columns='assetCode', values='open')
volume = market_train_df.pivot(index='time', columns='assetCode', values='volume')

returns = (close - close.shift()) / close
# returns_shifted_sma = returns.shift().rolling(window=20).mean()
returns_shifted_sm_std = returns.shift().rolling(window=20).std()

volume_change = (volume - volume.shift()) / volume
volume_shifted_sm_std = volume_change.shift().rolling(window=20).std()
surprise_factor = 1.5

return_signal = returns > (returns_shifted_sm_std * surprise_factor)
ret_and_vol_signal = (return_signal & (volume_change > volume_shifted_sm_std))

ret_and_vol_signal.index = list(map(lambda x: pd.to_datetime(x.date()),ret_and_vol_signal.index))
def filter_signal(signal_series, number_of_signal):
    base_case = number_of_signal * [False] + signal_series.tolist()  
    n = 0        
    while n < len(signal_series):
        if base_case[n+number_of_signal] and not sum(base_case[n:n+number_of_signal]):
            base_case[n+number_of_signal] = True
        else:
            base_case[n+number_of_signal] = False
        n += 1  
    return base_case[number_of_signal:]
nn = np.random.randint(len(return_signal))
'{} candidates by returns reduced to {} when also volume change is inspected. '.format(return_signal.iloc[:,nn].sum(), ret_and_vol_signal.iloc[:,nn].sum())
ret_and_vol_signal.tail(1)
def extract_assetCodes(df):
    dates, tickers, values = list(), list(), list()

    for ind, row in df.iterrows():
        date_data = pd.to_datetime(row.time.date())
        for part in row.assetCodes.split(','):
            particle = part.replace('{','').replace('}','').strip()
            dates.append(date_data)
            tickers.append(particle)
    values = [True] * len(dates)

    return pd.DataFrame({'dates':dates, 'tickers': tickers, 'values':values}).drop_duplicates().pivot(index='dates',columns='tickers',values='values')

novel_news_on_dates_by_assetcode = extract_assetCodes(novel_news)
novel_news_on_dates_by_assetcode.columns = list(map(lambda x:x[1:-1],novel_news_on_dates_by_assetcode.columns))
novel_news_on_dates_by_assetcode.tail(1)
news_dataframe = novel_news_on_dates_by_assetcode.loc[:,ret_and_vol_signal.columns].copy()
candidates = news_dataframe & ret_and_vol_signal
nn = np.random.randint(len(return_signal))
print('{} candidates found in case of {} symbol.'.format(candidates.iloc[:,nn].sum(), candidates.columns[nn]))

close__ = close.copy()
close__.index = list(map(lambda x: pd.to_datetime(x.date()),close__.index))

values_ = close__[candidates.columns[nn]]
markers = candidates[candidates.columns[nn]]*values_
markers = pd.DataFrame(data=np.where(markers == 0, np.nan, markers ),index=markers.index)

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(27,8))

ax1.plot(values_.index,values_,'b-')
ax1.plot(markers.index, markers,'r^')
input_data = input()
# input_data = 'ZQK'

for i, val in enumerate(candidates.columns):
    if input_data in val:
        nn = i
        
        print('{} candidates found in case of {} symbol.'.format(candidates.iloc[:,nn].sum(), candidates.columns[nn]))

        close__ = close.copy()
        close__.index = list(map(lambda x: pd.to_datetime(x.date()),close__.index))

        values_ = close__[candidates.columns[nn]]
        markers = candidates[candidates.columns[nn]]*values_
        markers = pd.DataFrame(data=np.where(markers == 0, np.nan, markers ),index=markers.index)

        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(27,8))

        ax1.plot(values_.index,values_,'b-')
        ax1.plot(markers.index, markers,'r^')
        
        break
else:
    print('No such asset found.')
# candidates[candidates.columns[nn]]
check_result = market_train_df[['time','assetCode','returnsOpenNextMktres10']]
check_result['time'] = check_result['time'].apply(lambda x: pd.to_datetime(x.date()))
check_result = check_result.pivot(index='time', columns='assetCode', values='returnsOpenNextMktres10')
result = check_result * candidates
log_result = np.where(np.abs(result) > 0, np.log(result), 0)
pd.DataFrame(data=log_result, index=result.index,columns=result.columns).sum().mean()



    

