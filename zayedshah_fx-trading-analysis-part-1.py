import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
trades_2015 = pd.read_excel('../input/Trades 2015.xlsx',sheet_name='Trades')
df = trades_2015
df.head()
df.info()
df.rename(columns={'Ticket #':'TicketId',
                   'Gross P/L':'GrossPnL','Net P/L':'NetPnL',
                   'Created By':'Account'},
          inplace=True)
df.head()
df.Account.unique()
df.Symbol.unique()
len(df.Symbol.unique())
df.agg({'GrossPnL':sum,
       'Rollover':sum,
       'NetPnL':sum})
plt.scatter(df.NetPnL.index, df.NetPnL)
plt.title('Net PnL Distribution\n2015', y=1.05)
plt.ylabel('Net PnL   (£)')
plt.xlabel('Trades', labelpad=10)
plt.scatter(df.NetPnL.index, df.NetPnL, alpha=0.5)
plt.title('Net PnL Distribution\n2015', y=1.05)
plt.ylabel('Net PnL   (£)')
plt.xlabel('Trades', labelpad=10)
ax = sns.boxplot(df.NetPnL)
ax.set_title('Net PnL Distribution\n2015')
ax.set_xlabel('Net PnL  (£)')             
df.NetPnL.describe()
df.sort_values('NetPnL').head(10).reset_index(drop=True)
df.sort_values('NetPnL', ascending=False).head(10).reset_index(drop=True)
df_ccy_maxPnL = pd.concat([df.sort_values('NetPnL').head(10), df.sort_values('NetPnL').tail(10)])

ax = df_ccy_maxPnL.plot.bar('Symbol','NetPnL',
                      color=np.sign(df_ccy_maxPnL.NetPnL).map({-1:'r', 1:'g'}),
                      legend=False,
                      alpha=0.7)
ax.set_title('Max PnL by Individual Trades\n2015', position=([0.5, 1.05]))
ax.set_ylabel('Net PnL   (£)')
ax.set_xlabel('Markets', labelpad=10)
plt.savefig('MaxPnL.png', bbox_inches='tight')
fig, ax = plt.subplots(figsize=(13, 5))
sns.boxplot(data=df, x='Symbol', y='NetPnL', ax=ax)
plt.xticks(rotation=90)
plt.title('Net PnL distribution by Markets\n2015', y=1.02)
plt.ylabel('Net PnL   (£)')
plt.xlabel('Markets', labelpad=15)
df_ccyPnL = df.groupby('Symbol', as_index=False)['NetPnL'].sum()
df_ccyPnL.sort_values('NetPnL').head(10).reset_index(drop=True)
df_ccyPnL.sort_values('NetPnL', ascending=False).head(10).reset_index(drop=True)
df_ccyPnL_Sum = pd.concat([df_ccyPnL.sort_values('NetPnL').head(10), df_ccyPnL.sort_values('NetPnL').tail(10)])

ax = df_ccyPnL_Sum.plot.bar('Symbol','NetPnL',
                        color=np.sign(df_ccyPnL_Sum.NetPnL).map({-1:'r', 1:'g'}),
                        legend=False,
                        alpha=0.7)
ax.set_title('Aggregate PnL by Markets\n2015', position=([0.5, 1.05]))
ax.set_ylabel('Net PnL   (£)')
ax.set_xlabel('Markets', labelpad=10)
df.Close = pd.to_datetime(df.Close)
df['Month'] = df.Close.map(lambda x: x.strftime('%b'))
df['Month'] = df.Month.astype('category', categories=
                           ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec'])

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='Month', y='NetPnL', ax=ax)
plt.xticks(rotation=90)
plt.title('Net PnL distribution by Month\n2015', y=1.02)
plt.ylabel('Net PnL   (£)')
plt.xlabel('Month', labelpad=15)

df_monthPnL = df.groupby('Month', as_index=False)['NetPnL'].sum()
df_monthPnL
def colorMap(x):
    return 'color: %s' % ('red' if x < 0 else 'white' if x == 0 else 'green')
    
df_monthPnL.style.applymap(colorMap, subset='NetPnL')
ax = df_monthPnL.plot.bar('Month','NetPnL',
                   color=np.sign(df_monthPnL.NetPnL).map({-1:'r', 1:'g'}),
                   alpha=0.7,
                   legend=False)
ax.set_title('Monthly PnL\n2015', position=([0.5, 1.05]))
ax.set_ylabel('Net PnL   (£)')
ax.set_xlabel('Month', labelpad=10)
df_monthPnL_sorted = df_monthPnL.sort_values('NetPnL').reset_index(drop=True)

df_monthPnL_sorted.style.applymap(colorMap, subset='NetPnL')
ax = df_monthPnL_sorted.plot.bar('Month','NetPnL',
                   color=np.sign(df_monthPnL_sorted.NetPnL).map({-1:'r', 1:'g'}),
                   alpha=0.7,
                   legend=False)
ax.set_title('Monthly PnL\n2015', position=([0.5, 1.05]))
ax.set_ylabel('Net PnL   (£)')
ax.set_xlabel('Month', labelpad=10)
df_pt_ccy_month = df.pivot_table(values='NetPnL', index='Symbol', columns='Month', aggfunc=sum,
                      margins=True, margins_name='Total').fillna(0)
df_pt_ccy_month.drop('Total', axis=0, inplace=True)
cols = df_pt_ccy_month.columns

df_ccy_losses = df_pt_ccy_month.sort_values('Total').head(10)
df_ccy_losses.style.applymap(colorMap, subset=cols)
df_ccy_profits = df_pt_ccy_month.sort_values('Total', ascending=False).head(10)
df_ccy_profits.style.applymap(colorMap, subset=cols)