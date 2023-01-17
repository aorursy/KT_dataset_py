import pandas as pd

filename = "/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"

df = pd.read_csv(filename)

df.head()
df['Timestamp']= pd.to_datetime(df['Timestamp'], unit='s') 
df = df.dropna()
df = df.rename(columns={"Timestamp": "Date", "Weighted_Price": "USD"})
df = df.set_index('Date')
df = df.resample('D').mean()
df = df.filter(['Date','USD'])
from itertools import combinations

df2 = pd.DataFrame(combinations(df.index, 2), columns=['dateBuy','dateSell'])

df2.head()
df['BTC'] = 1 / df
import time

start_time = time.time()

def f(x):

   return df[x['dateBuy']:x['dateSell']]['BTC'].sum()



df2[0:10000]['sum'] = df2[0:10000].apply(f, axis=1)

# df2.head()



print("--- %s minutes ---" % (4600000/(10000/(time.time() - start_time))/60))
df2[0:3]
df[0:4]