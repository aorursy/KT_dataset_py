import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
trades_2015 = pd.read_excel('../input/trades-2015/Trades 2015.xlsx')
trades_2016 = pd.read_excel('../input/trades-2016/Trades 2016.xlsx')
df = pd.concat([trades_2015, trades_2016])
df.info()
len(trades_2015) + len(trades_2016)
df.head()
df['Close'] = pd.to_datetime(df.Close)
df.info()
df['Month'] = df['Close'].map(lambda x: x.strftime('%b'))
df.head()
df['Year'] = df.Close.map(lambda x: x.strftime('%Y'))
df.head()
df.Year.unique()
df.Month.unique()
df.Month = df.Month.astype('category', categories=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
df.rename(columns={'Net P/L':'NetPnL','Gross P/L':'GrossPnL','Created By':'Account','Ticket #':'TicketId'},
         inplace=True)
df.head()
df.groupby('Year', as_index=False)['NetPnL'].sum()
df_t = df.groupby('Year', as_index=False)['NetPnL'].sum() 
# df_t is short for dataframe_temp

df_t['AUM'] = 0
df_t.loc[0,'AUM'] = 11000
df_t.loc[1, 'AUM'] = df_t.loc[0, 'AUM'] - np.abs(df_t.loc[0, 'NetPnL'])
df_t['Performance'] = 0
df_t.Performance = round(100 * df_t.NetPnL / df_t.AUM, 2)
df_t.Performance = df_t.Performance.map(lambda x: '{:.2f}%'.format(x))
df_t = df_t[['Year','AUM','NetPnL','Performance']]
df_t.set_index('Year', inplace=True)
df_t
df_15 = df[df.Year == '2015']
df_15.head()
df_16 = df[df.Year == '2016']
df_16.head()
df_15.info()
df_16.info()
f, (a0, a1) = plt.subplots(1, 2, sharey=True, figsize=(12,4))
f.suptitle('Net PnL Distribution')

a0.scatter(df_15.NetPnL.index, df_15.NetPnL)
a1.scatter(df_16.NetPnL.index, df_16.NetPnL)

a0.set_title('2015')
a0.set_xlabel('Trades', labelpad=10)
a0.set_ylabel('Net PnL   (£)')

a1.set_title('2016')
a1.set_xlabel('Trades', labelpad=10)
f, (a0, a1) = plt.subplots(1, 2, sharey=True, figsize=(12,4))
f.suptitle('Net PnL Distribution')

a0.scatter(df_15.NetPnL.index, df_15.NetPnL, alpha=0.5)
a1.scatter(df_16.NetPnL.index, df_16.NetPnL, alpha=0.5)

a0.set_title('2015')
a0.set_xlabel('Trades', labelpad=10)
a0.set_ylabel('Net PnL   (£)')

a1.set_title('2016')
a1.set_xlabel('Trades', labelpad=10)
f, (a0, a1) = plt.subplots(2, 1, sharex=True, figsize=(6,6))
f.suptitle('Net PnL Distribution')

sns.boxplot(df_15.NetPnL, ax=a0)
sns.boxplot(df_16.NetPnL, ax=a1)

a0.set_title('2015')
a0.set_xlabel('')

a1.set_title('2016')
a1.set_xlabel('Net PnL   (£)', labelpad=10)
df_pt_month_yr = df.pivot_table(values='NetPnL', aggfunc=sum,
                                index='Month', columns='Year')

def colorMap(x):
    return 'color: %s' % ('red' if x < 0 else 'white' if x == 0 else 'green')

df_pt_month_yr.style.applymap(colorMap, subset=['2015','2016'])
df_pt_month_yr.describe().round(2)
df_pt_month_yr_tmp = df_pt_month_yr.copy()
df_pt_month_yr_tmp.columns.name = None
df_pt_month_yr_tmp.reset_index(inplace=True)

f, (a0, a1) = plt.subplots(1, 2, sharey=True, figsize=(10,4))
f.suptitle('Monthly PnL')

def colorMapGraph(df, col):
    return np.sign(df[col]).map({-1:'r',1:'g'})

def plotBar(year, ax):
    df_pt_month_yr_tmp.plot.bar('Month',year, ax=ax,
                            color=colorMapGraph(df_pt_month_yr_tmp, year),
                            alpha=0.7,
                            legend=False)
    ax.set_title(year)
    ax.set_xlabel('Month', labelpad=10)

plotBar('2015', a0)
plotBar('2016', a1)

a0.set_ylabel('Net PnL   (£)')
df_15_t = df_15.groupby(['Month'], as_index=False)['NetPnL'].sum().sort_values('NetPnL').reset_index(drop=True)
df_16_t = df_16.groupby(['Month'], as_index=False)['NetPnL'].sum().sort_values('NetPnL').reset_index(drop=True)
df_ct = pd.concat([df_15_t, df_16_t], axis=1, keys=['2015','2016'])
df_ct.style.applymap(colorMap, subset=[('2015','NetPnL'),(('2016','NetPnL'))])
f, (a0, a1) = plt.subplots(1, 2, sharey=True, figsize=(10,4))
f.suptitle('Monthly PnL Sorted')

def plot_MonthlyPnL_Sorted(df, year, ax):
    df.plot.bar('Month','NetPnL', ax=ax,
                color=colorMapGraph(df, 'NetPnL'),
                alpha=0.7,
                legend=False)
    ax.set_title(year)
    ax.set_xlabel('Month', labelpad=10)

plot_MonthlyPnL_Sorted(df_15_t, '2015', a0)
plot_MonthlyPnL_Sorted(df_16_t, '2016', a1)

a0.set_ylabel('Net PnL   (£)')
df_15_MaxPnL = pd.concat([df_15.sort_values('NetPnL').head(10),
                          df_15.sort_values('NetPnL').tail(10)])

df_16_MaxPnL = pd.concat([df_16.sort_values('NetPnL').head(10),
                          df_16.sort_values('NetPnL').tail(10)])

df_15_t1 = df_15_MaxPnL[['Symbol','NetPnL']].reset_index(drop=True) 
df_16_t1 = df_16_MaxPnL[['Symbol','NetPnL']].reset_index(drop=True)

df_ct1 = pd.concat([df_15_t1, df_16_t1], axis=1, keys=['2015', '2016']) 
df_ct1.rename(columns={'Symbol':'Markets'}, inplace=True)

df_ct1.style.applymap(colorMap, subset=[('2015','NetPnL'),(('2016','NetPnL'))])
f, (a0, a1) = plt.subplots(1, 2, sharey=True, figsize=(12,4))
f.suptitle('Max PnL by Individual Trades')

def plotBar_PnL(df, year, ax):
    df.plot.bar('Symbol','NetPnL', ax=ax,
                            color=colorMapGraph(df, 'NetPnL'),
                            alpha=0.7,
                            legend=False)
    ax.set_title(year)
    ax.set_xlabel('Markets', labelpad=10)

plotBar_PnL(df_15_MaxPnL, '2015', a0)
plotBar_PnL(df_16_MaxPnL, '2016', a1)

a0.set_ylabel('Net PnL   (£)')

plt.savefig('MaxPnL.png', bbox_inches='tight')
def concat_aggPnL(df):
    return pd.concat([
        df.groupby('Symbol', as_index=False)['NetPnL'].sum().sort_values('NetPnL').head(10),
        df.groupby('Symbol', as_index=False)['NetPnL'].sum().sort_values('NetPnL').tail(10)])
    
df_15_aggPnL = concat_aggPnL(df_15)
df_16_aggPnL = concat_aggPnL(df_16)

df_15_t2 = df_15_aggPnL.reset_index(drop=True)
df_16_t2 = df_16_aggPnL.reset_index(drop=True)

df_ct2 = pd.concat([df_15_t2, df_16_t2], axis=1, keys=['2015', '2016']) 
df_ct2.rename(columns={'Symbol':'Markets'}, inplace=True)

df_ct2.style.applymap(colorMap, subset=[('2015','NetPnL'),(('2016','NetPnL'))])
f, (a0, a1) = plt.subplots(1, 2, sharey=True, figsize=(12,4))
f.suptitle('Aggregate PnL by Markets')

plotBar_PnL(df_15_aggPnL, '2015', a0)
plotBar_PnL(df_16_aggPnL, '2016', a1)

a0.set_ylabel('Net PnL   (£)')