!pip install yahoofinance > /dev/null
import yahoofinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
%matplotlib inline
q = yf.HistoricalPrices('qqq', '2019-10-21', '2020-10-21')
df_qqq = q.to_dfs()['Historical Prices']
df_qqq['Name'] = 'QQQ'

df_qqq.head()
v = yf.HistoricalPrices('voo', '2019-10-21', '2020-10-21')
df_voo = v.to_dfs()['Historical Prices']
df_voo['Name'] = 'VOO'

df_voo.head()
x = yf.HistoricalPrices('xle', '2019-10-21', '2020-10-21')
df_xle = x.to_dfs()['Historical Prices']
df_xle['Name'] = 'XLE'

df_xle.head()
j = yf.HistoricalPrices('jets', '2019-10-21', '2020-10-21')
df_jets = j.to_dfs()['Historical Prices']
df_jets['Name'] = 'JETS'

df_jets.head()
df_qqq.info()
df_qqq.describe()
etfs = [df_qqq, df_voo, df_xle, df_jets]
etfs_names = ['QQQ', 'VOO', 'XLE', 'JETS']
fig, ax = plt.subplots(figsize=(12, 10))

for i, etf in enumerate(etfs, 1):
    plt.subplot(2, 2, i)
    etf['Adj Close'].plot(linewidth=2)
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"{etf['Name'][i - 1]}")
for i, etf in enumerate(etfs):
    print("perforance of", etfs_names[i], "through the period: ", (etf['Adj Close'][-1] - etf['Adj Close'][0])*100/etf['Adj Close'][0], "percent")
for etf in etfs:
    etf['Daily Return'] = etf['Adj Close'].pct_change()
plt.figure(figsize=(12, 10))

for i, etf in enumerate(etfs, 1):
    plt.subplot(2, 2, i)
    sns.distplot(etf['Daily Return'].dropna(), bins=100, color='purple')
    plt.ylabel('Daily Return')
    plt.title(f"{etf['Name'][i - 1]}")
for i, etf in enumerate(etfs):
    print("average daily return of", etfs_names[i], "is", etf['Daily Return'].mean(),
          "percent, with standard deviation of", etf['Daily Return'].std())
close_prices = pd.DataFrame([df_qqq['Adj Close'], df_voo['Adj Close'], df_xle['Adj Close'], df_jets['Adj Close']])

close_prices = close_prices.transpose()

fig, ax = plt.subplots(figsize=(5,5))

x_axis_labels = [i for i in etfs_names]
y_axis_labels = [i for i in etfs_names]

corr = close_prices.corr()

sns.heatmap(corr, annot=True, cmap='YlGnBu', ax = ax,
            xticklabels=x_axis_labels, yticklabels=y_axis_labels)

for etf in etfs:
    
    etf['MA20'] = etf['Adj Close'].rolling(window=20).mean()
    etf['20dSTD'] = etf['Adj Close'].rolling(window=20).std() 

    etf['Upper'] = etf['MA20'] + (etf['20dSTD'] * 2)
    etf['Lower'] = etf['MA20'] - (etf['20dSTD'] * 2)
plt.figure(figsize=(12, 10))

for i, etf in enumerate(etfs, 1):
    plt.subplot(2, 2, i)
    etf['Adj Close'].plot(linewidth=2)
    etf['Lower'].plot(linewidth=2)
    etf['MA20'].plot(linewidth=2)
    etf['Upper'].plot(linewidth=2)
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"{etf['Name'][i - 1]}")
    plt.legend()