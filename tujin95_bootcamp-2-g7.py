%matplotlib inline 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
df_crsp = pd.read_csv('../input/debondt-data/monthly.csv')
df_crsp = pd.read_csv('../input/week2-data/monthly.csv') 
df_crsp = pd.read_csv('../input/debondt-data/monthly.csv', na_values=['C'], parse_dates=['DATE']) 
df_crsp = pd.read_csv('../input/week2-data/monthly.csv', na_values=['C'], parse_dates=['DATE'])

# The output from WRDS returns a mixed of small and large cap column names. Let's standardize everything in small caps.
cols = df_crsp.columns
print(cols)
print(type(cols))

# List comprehension [expression for variable in iterable]
df_crsp.columns = [c.lower() for c in cols]

# Sanitize dataset

# Drop observations with missing returns
df_crsp = df_crsp[df_crsp.ret.notnull()]

# Take the absolute value of the price
df_crsp['prc'] = np.abs(df_crsp['prc'])

# Set the index (to select easily on date)
df_crsp = df_crsp.set_index('date')
df_crsp = df_crsp.sort_index()

# Compute continuously compounded returns (i.e. log returns).
df_crsp['lret'] = np.log(1 + df_crsp['ret'])
df_crsp['lvwretd'] = np.log(1 + df_crsp['vwretd'])
df_crsp['lewretd'] = np.log(1 + df_crsp['ewretd'])

# Compute the market cap (shrout = shares outstanding)
df_crsp['size'] = df_crsp['shrout'] * df_crsp['prc']
# Parameters
form_period = 36 # 36 Formation period, in month
start_date = '1933-01-01' 
end_date = '2000-01-01' 

# Get the dates of portfolio formation.
# The frequency tells how far apart to put the dates.
# 'M' stand for month, 'MS' is for month start, to make sure we 
# have first day of the month. It needs to be a string, so we convert
# our numbers to string. 
dates = pd.date_range(start=start_date, end=end_date, freq=str(np.int(form_period)) + 'M')
dates
form_period = 36 # 36 Formation period, in month
n_stocks = 35  # Number of stocks in the top and bottom performance
benchmark = 'vwretd' # Benchmark market return to use ('vwretd' or 'ewretd')
def compute_performance_portfolios(date, df, form_period=36, n_stocks=35,
                                   benchmark='vwretd'):
    beg_dt = date - pd.offsets.MonthBegin(1) * form_period

    # Select obs for the formation period
    crsp_t = df[beg_dt:date].copy()
    
    # We only want to keep stocks that are there during the full formation window
    crsp_t['N'] = crsp_t.groupby(['permno'])['permno'].transform('count')
    # Filter on number of observations. We only keep sotcks for which we have returns
    # over the full observation period.
    crsp_t = crsp_t[crsp_t['N'] >= form_period]
    
    # Now for each stock we want to compute the full period return. Easy with log returns, just sum up!
    stock_ret = crsp_t.groupby('permno')[['lret', 'lvwretd', 'lewretd']].sum()
    
    # Next compute excess returns based on the chosen index.
    # Note that since the benchmark is the same for all stocks, we could use
    # actual returns for ranking purposes.
    stock_ret['lexret'] = stock_ret['lret'] - stock_ret['l' + benchmark]
    
    # Now rankings.
    stock_ret['rank_asc'] = stock_ret['lexret'].rank() # (1 = worst return)
    stock_ret['rank_inv'] = stock_ret['lexret'].rank(ascending=False) # (1= best return)
    
    # Assign stock to top or bottom portfolio
    top_portfolio = stock_ret[stock_ret.rank_inv <= n_stocks].reset_index()[['permno', 'lexret']]
    bottom_portfolio = stock_ret[stock_ret.rank_asc <= n_stocks].reset_index()[['permno', 'lexret']]
    
    return (bottom_portfolio, top_portfolio)
portfolios = {}
for date in dates:
    portfolios[date] = compute_performance_portfolios(date, df_crsp, benchmark='vwretd')
portfolios[date][0].head() #bottom portfolio
hold_period = 36 # Holding period, in months
benchmark = 'vwretd' # 'vwretd' or 'ewretd'
weighting = 'vw' # 'vw' or 'ew'
def compute_holding_returns(date, portfolio, df, benchmark='vwretd', weighting='vw', hold_period=36):
    portfolio = portfolio.copy()
    end_dt = date + pd.offsets.MonthBegin(1) * hold_period
    # Select obs for the formation period
    crsp_t2 = df[date:end_dt].copy()
    # Merge with stocks in portfolios, to keep only those stocks
    crsp_t2 = pd.merge(crsp_t2.reset_index(), portfolio, on=['permno'])
    crsp_t2
    

    # Get the dates in the dataset.
    pairs_t2 = [{'date': d, 'permno': p} for d in crsp_t2['date'].unique() for p in portfolio['permno'].unique()]
    pairs_t2 = pd.DataFrame(pairs_t2)
    pairs_t2
    crsp_t2 = pd.merge(crsp_t2, pairs_t2, how='outer', on=['permno', 'date'])
    ret_cols = ['ret', 'vwretd', 'ewretd', 'lvwretd','lewretd', 'lret', 'lexret']
    crsp_t2[ret_cols] = crsp_t2[ret_cols].fillna(0.0)
    
    # Now we want the return up to each point in time
    crsp_t2['lcumret'] = crsp_t2.groupby('permno')['lret'].cumsum()
    crsp_t2['lcum' + benchmark] = crsp_t2.groupby('permno')['l' + benchmark].cumsum()

    # At each point in time, the return of the portfolio will be the 
    # cumulative return of each component weighted by the initial weight.
    # Note that here we need the simple return average, not log return.
    crsp_t2['cumret'] = np.exp(crsp_t2['lcumret']) - 1
    crsp_t2['cum' + benchmark] = np.exp(crsp_t2['lcum' + benchmark]) - 1

    # Add weights, equal weighted is easy.
    portfolio['ew'] = 1 / len(portfolio)

    # For value-weighted, need to get size as of formation date.
    portfolio['date'] = date
    weights = pd.merge_asof(portfolio, df[['permno', 'size']],
                            by='permno',
                            left_on='date',
                            right_index=True)
    weights['vw'] = weights['size'] / weights['size'].sum()

    del weights['lexret']
    del weights['date']
    del weights['size']
    
    # Now merge back with returns
    crsp_t2 = pd.merge(crsp_t2, weights, on='permno')
    
    # Now compute the weighted cumulative return
    crsp_t2['wcumret'] = crsp_t2[weighting] * crsp_t2['cumret']
    crsp_t2['wcum' + benchmark] = crsp_t2[weighting] * crsp_t2['cum' + benchmark]

    portfolio_ret = crsp_t2.groupby(['date'])[['wcumret', 'wcum' + benchmark]].sum()
    
    # Count the months
    portfolio_ret = portfolio_ret.reset_index()
    portfolio_ret['months'] = portfolio_ret.index.values + 1
    
    portfolio_ret['exret'] = portfolio_ret['wcumret'] - portfolio_ret['wcum' + benchmark]
    
    return portfolio_ret
bottom_portfolio_ret = []
top_portfolio_ret = []

for date in dates:
    bottom_portfolio_ret.append(compute_holding_returns(date, portfolios[date][0], df_crsp, benchmark='vwretd', weighting='vw'))
    top_portfolio_ret.append(compute_holding_returns(date, portfolios[date][1], df_crsp, benchmark='vwretd', weighting='vw'))

bottom_portfolio_ret = pd.concat(bottom_portfolio_ret)
top_portfolio_ret = pd.concat(top_portfolio_ret)
# Only pre 1980
plt.figure(figsize=(10,10))

ax = bottom_portfolio_ret.set_index('date')[:'1980'].groupby('months')['exret'].mean().plot(label='Past losers')
top_portfolio_ret.set_index('date')[:'1980'].groupby('months')['exret'].mean().plot(ax=ax, label='Past winners')
ax.legend()
ax.axhline(y=0,  color='black', alpha=0.5, linestyle=':')
ax.axvline(x=12, color='black', alpha=0.5, linestyle='-')
ax.axvline(x=24, color='black', alpha=0.5, linestyle='-')


#Top portfolio test

tset, pval = stats.ttest_1samp(top_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'],0)

print('Mean Excess Return for Top Performing Funds ' + str(top_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean()))
print('P-values ' + str(pval))
if pval < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
#Bottom Portfolio Test

tset, pval = stats.ttest_1samp(bottom_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'],0)

print('Mean Excess Return for Bottom Performing Funds ' + str(bottom_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean()))
print('P-values ' + str(pval))
if pval < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
#Ret Diff test

tset, pval = stats.ttest_ind(top_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'],
                             bottom_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'])

print('Mean Excess Return for Top Performing Funds ' + str(top_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean()))
print('Mean Excess Return for Bottom Performing Funds ' + str(bottom_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean()))
print('Mean Difference Bottom Minus Top in Excess Return ' + 
      str(bottom_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean() -                                                        
          top_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean()))
print('P-values ' + str(pval))
if pval < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
#setup test
benchmark='vwretd'
n_stocks=35
weighting='vw'
dates=dates
df_test=df_crsp
form_period=36
hold_per=36
s_year='1933'
e_year='1980'

def test(s_year,e_year,hold_per,form_period,df_test,dates,weighting,n_stocks,benchmark):
    #setup portfolio
    portfolios_test = {}
    for date in dates:
        portfolios_test[date] = compute_performance_portfolios(date, df_test, form_period=form_period, n_stocks=n_stocks,benchmark=benchmark)

    #portfolio returns
    bottom_portfolio_ret_test = []
    top_portfolio_ret_test = []
    for date in dates:
        bottom_portfolio_ret_test.append(compute_holding_returns(date, portfolios_test[date][0], df_test, benchmark=benchmark, weighting=weighting, hold_period=hold_per))
        top_portfolio_ret_test.append(compute_holding_returns(date, portfolios_test[date][1], df_test, benchmark=benchmark, weighting=weighting ,hold_period=hold_per))

    # join to single df
    bottom_portfolio_ret_test = pd.concat(bottom_portfolio_ret_test)
    top_portfolio_ret_test = pd.concat(top_portfolio_ret_test)

    #Analysis
    tset, pval = stats.ttest_ind(top_portfolio_ret_test[top_portfolio_ret_test['months']==hold_per].set_index('date')[s_year:e_year]['exret'],
                                 bottom_portfolio_ret_test[top_portfolio_ret_test['months']==hold_per].set_index('date')[s_year:e_year]['exret'])
    
    top_ret=top_portfolio_ret_test[top_portfolio_ret_test['months']==hold_per].set_index('date')[s_year:e_year]['exret'].mean()
    bot_ret = bottom_portfolio_ret_test[top_portfolio_ret_test['months']==hold_per].set_index('date')[s_year:e_year]['exret'].mean()
    bot_top_ret = bottom_portfolio_ret_test[top_portfolio_ret_test['months']==hold_per].set_index('date')[s_year:e_year]['exret'].mean() - top_portfolio_ret_test[top_portfolio_ret_test['months']==hold_per].set_index('date')[s_year:e_year]['exret'].mean()
    if pval< 0.05:
        outcome = "Reject"
    else:
        outcome = "Accept"
    return [top_ret,bot_ret,bot_top_ret,pval,outcome]

def plot(s_year,e_year,hold_per,form_period,df_test,dates,weighting,n_stocks,benchmark): 
    #setup portfolio
    portfolios_test = {}
    for date in dates:
        portfolios_test[date] = compute_performance_portfolios(date, df_test, form_period=form_period, n_stocks=n_stocks,benchmark=benchmark)

    #portfolio returns
    bottom_portfolio_ret_test = []
    top_portfolio_ret_test = []
    for date in dates:
        bottom_portfolio_ret_test.append(compute_holding_returns(date, portfolios_test[date][0], df_test, benchmark=benchmark, weighting=weighting, hold_period=hold_per))
        top_portfolio_ret_test.append(compute_holding_returns(date, portfolios_test[date][1], df_test, benchmark=benchmark, weighting=weighting ,hold_period=hold_per))
    # join to single df
    bottom_portfolio_ret_test = pd.concat(bottom_portfolio_ret_test)
    top_portfolio_ret_test = pd.concat(top_portfolio_ret_test)
    

    # plot
    
    #ax.set_title("n="+str(n))
    
    ax = bottom_portfolio_ret_test.set_index('date')[s_year:e_year].groupby('months')['exret'].mean().plot(label='Past losers, n=' + str(n_stocks))
    top_portfolio_ret_test.set_index('date')[s_year:e_year].groupby('months')['exret'].mean().plot(ax=ax, label='Past winners, n=' + str(n_stocks))
    ax.legend()
    ax.axhline(y=0,  color='black', alpha=0.5, linestyle=':')
    #ax.axvline(x=12, color='black', alpha=0.5, linestyle='-')
    #ax.axvline(x=24, color='black', alpha=0.5, linestyle='-')
    
fig, ax = plt.subplots(figsize=(10, 10))

for n_stocks in [10,35]:
    plot(s_year,e_year,hold_per,form_period,df_test,dates,weighting,n_stocks,benchmark)
#Plot: changing n_stocks (reflect definitions of winners/losers)

def test_n(n):
    portfolios = {}
    for date in dates:
        portfolios[date] = compute_performance_portfolios(date, df_crsp, benchmark='vwretd',n_stocks=n)
    
    bottom_portfolio_ret = []
    top_portfolio_ret = []

    for date in dates:
        bottom_portfolio_ret.append(compute_holding_returns(date, portfolios[date][0], df_crsp, benchmark='vwretd', weighting='vw'))
        top_portfolio_ret.append(compute_holding_returns(date, portfolios[date][1], df_crsp, benchmark='vwretd', weighting='vw'))

    bottom_portfolio_ret = pd.concat(bottom_portfolio_ret)
    top_portfolio_ret = pd.concat(top_portfolio_ret)
    
    
    ax = bottom_portfolio_ret.groupby('months')['exret'].mean().plot(label='Past losers')
    top_portfolio_ret.groupby('months')['exret'].mean().plot(ax=ax, label='Past winners')
    
    plt.figure(figsize=(5,5))
    ax.set_title("n="+str(n))
    ax.axhline(y=0,  color='black', alpha=0.5, linestyle=':')
    ax.axvline(x=12, color='black', alpha=0.5, linestyle='-')
    ax.axvline(x=24, color='black', alpha=0.5, linestyle='-')
    ax.legend()
    
    
for n in [10]:
    test_n(n)

tset, pval = stats.ttest_ind(top_portfolio_ret[top_portfolio_ret['months']==36]['exret'], bottom_portfolio_ret[top_portfolio_ret['months']==36]['exret'])
#changing becnhmark market return to equally weighted index ('ewretd')

#changing stock weighting to equally-weighted ('ew')

#chaning formation month from Jan, Mar, June, Sept, Dec

#changing formation periods to 12,24,36,48,60 months
# will take some time to run, dont close the notebook
# Original Output
print('Original Output')
print('-----------------------------------------------------------------------------')
out_og =test(s_year,e_year,hold_per,form_period,df_crsp,dates,weighting,n_stocks,benchmark)
top_ret, bot_ret, bot_top_ret, pval, outcome = out_og
print('Mean Excess Return for Top Performing Funds ' + str(top_ret))
print('Mean Excess Return for Bottom Performing Funds ' + str(bot_ret))
print('Mean Difference Bottom Minus Top in Excess Return ' + 
      str(bot_top_ret))
print('P-values ' + str(pval))
if pval < 0.05:    # alpha value is 0.05 or 5%
    print("we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
print('-----------------------------------------------------------------------------')
print('\n')


# winners/loosers sensitivity
print('winners/loosers sensitivity')
print('-----------------------------------------------------------------------------')
df_out=pd.DataFrame()
for i in range(15,65,10):
    out = test(s_year,e_year,hold_per,form_period,df_crsp,dates,weighting,i,benchmark)
    out.insert(0,i)
    df_out=df_out.append(pd.Series(out,index=['n_stocks','1.Top Ex.Ret.','2.Bottom Ex.Ret.','3.Bottom-Top ','4.Pval','5.Outcome']),ignore_index=True, sort=True)
print(df_out.set_index('n_stocks').round(3))
print('-----------------------------------------------------------------------------')
print('\n')


# Benchmark sensitivity
print('Benchmark sensitivity')
print('-----------------------------------------------------------------------------')
df_out=pd.DataFrame()
out = test(s_year,e_year,hold_per,form_period,df_crsp,dates,weighting,n_stocks,'ewretd')
out.insert(0,'equal wtd.')
df_out=df_out.append(pd.Series(out,index=['Benchmark','1.Top Ex.Ret.','2.Bottom Ex.Ret.','3.Bottom-Top ','4.Pval','5.Outcome']),ignore_index=True, sort=True)
out_og_temp=out_og.copy()
out_og_temp.insert(0,'value wtd.')
df_out=df_out.append(pd.Series(out_og_temp,index=['Benchmark','1.Top Ex.Ret.','2.Bottom Ex.Ret.','3.Bottom-Top ','4.Pval','5.Outcome']),ignore_index=True, sort=True)
print(df_out.set_index('Benchmark').round(3))
print('-----------------------------------------------------------------------------')
print('\n')


# Weighting sensitivity
print('Weighting sensitivity')
print('-----------------------------------------------------------------------------')
df_out=pd.DataFrame()
out = test(s_year,e_year,hold_per,form_period,df_crsp,dates,'ew',n_stocks,benchmark)
out.insert(0,'equal wtd.')
df_out=df_out.append(pd.Series(out,index=['Weighted','1.Top Ex.Ret.','2.Bottom Ex.Ret.','3.Bottom-Top ','4.Pval','5.Outcome']),ignore_index=True, sort=True)
out_og_temp=out_og.copy()
out_og_temp.insert(0,'value wtd.')
df_out=df_out.append(pd.Series(out_og_temp,index=['Weighted','1.Top Ex.Ret.','2.Bottom Ex.Ret.','3.Bottom-Top ','4.Pval','5.Outcome']),ignore_index=True, sort=True)
print(df_out.set_index('Weighted').round(3))
print('-----------------------------------------------------------------------------')
print('\n')


# Formation Month sensitivity
print('Formation Month sensitivity')
print('-----------------------------------------------------------------------------')
df_out=pd.DataFrame()
for i in range(1,13,2):
    month=str(i)
    if len(month)<2:
        month ='0'+month
    dates_temp=pd.date_range(start='1933-'+month+'-01', end=end_date, freq=str(np.int(form_period)) + 'M')
    out = test(s_year,e_year,hold_per,form_period,df_crsp,dates_temp,weighting,n_stocks,benchmark)
    out.insert(0,i)
    df_out=df_out.append(pd.Series(out,index=['Months','1.Top Ex.Ret.','2.Bottom Ex.Ret.','3.Bottom-Top ','4.Pval','5.Outcome']),ignore_index=True, sort=True)
print(df_out.set_index('Months').round(3))
print('-----------------------------------------------------------------------------')
print('\n')


# Formation Period sensitivity
print('Formation Period sensitivity')
print('-----------------------------------------------------------------------------')
df_out=pd.DataFrame()
for i in range(12,84,12):
    dates_temp=pd.date_range(start=start_date, end=end_date, freq=str(np.int(i)) + 'M')
    out = test(s_year,e_year,hold_per,i,df_crsp,dates_temp,weighting,n_stocks,benchmark)
    out.insert(0,i)
    df_out=df_out.append(pd.Series(out,index=['Period(m)','1.Top Ex.Ret.','2.Bottom Ex.Ret.','3.Bottom-Top ','4.Pval','5.Outcome']),ignore_index=True, sort=True)
print(df_out.set_index('Period(m)').round(3))
print('-----------------------------------------------------------------------------')
print('\n')
print('Post 1980')
print('-----------------------------------------------------------------------------')
out =test('1980','2000',hold_per,form_period,df_crsp,dates,weighting,n_stocks,benchmark)
top_ret, bot_ret, bot_top_ret, pval, outcome = out
print('Mean Excess Return for Top Performing Funds ' + str(top_ret))
print('Mean Excess Return for Bottom Performing Funds ' + str(bot_ret))
print('Mean Difference Bottom Minus Top in Excess Return ' + 
      str(bot_top_ret))
print('P-values ' + str(pval))
if pval < 0.05:    # alpha value is 0.05 or 5%
    print("we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
print('-----------------------------------------------------------------------------')

plot('1980','2000',hold_per,form_period,df_crsp,dates,weighting,n_stocks,benchmark) 
#import data from yahoo finance

#run below commented line "!pip..." to install Yfinance package
#!pip install yfinance
import yfinance as yf
tickers = ["UCO","GOEX", "ARKW", "SBIO", "SPY", "QQQ", "EEM", "EUSA", "GLD", "THD",
          "AAXJ","EWS","ACWI","VIXY", "BOND", "VGLT", "EMB", "SUSA", "EWM", "EIDO",
          "ICLN", "VDE", "VDC", "YINN", "YANG", "CQQQ", "CWI", "ESGU", "SPDW", "USMV",
          "BJK", "IAT", "VFH", "VNQ", "EWU", "EWQ", "EWG", "IJR","PEX","IGF",
          "VTI","VTV", "VB", "XLK","VYM","QUAL", "IWP", "XLU", "VHT","EWJ","XLC"]
yf_data = yf.download( tickers = tickers, period = "20y", interval = "1mo")
#yf_data = yf_data.fillna(method = 'ffill')
#data prep

# MSCI ACWI used as benchmark, as we have global ETFs
df=yf_data['Adj Close'].groupby(pd.Grouper(freq="M")).mean().reset_index()
df=df.fillna(method='ffill')
df=df.rename(columns={'Date':'date'})
df=df.melt(id_vars=["date"], var_name="permno", value_name="prc").set_index('date')
df['ret']=df.groupby('permno').pct_change(1)
df['ewretd']=df[df['permno']=='ACWI']['ret'] # 'ewretd'is replaced by MSCI ACWI as our benchmark
df['lret']= np.log(1 + df['ret'])
df['lewretd']=np.log(1+df['ewretd'])
df=df.dropna()

# not applicaple fields for this analysis filled with NAN to make use of above functions
df['shrout']=float('NaN') 
df['vwretd']=float('NaN')
df['lvwretd']=float('NaN')
df['size']=float('NaN')
df=df.sort_index()
df.head()
dates_tempx=pd.date_range(start='2011-01-01', end='2019-01-31', freq=str(np.int(12)) + 'M')

print('ETF Analysis')
print('-----------------------------------------------------------------------------')
out = test('2011','2019',12,12,df,dates_tempx,'ew',20,'ewretd')
top_ret, bot_ret, bot_top_ret, pval, outcome = out
print('Mean Excess Return for Top Performing Funds ' + str(top_ret))
print('Mean Excess Return for Bottom Performing Funds ' + str(bot_ret))
print('Mean Difference Bottom Minus Top in Excess Return ' + 
      str(bot_top_ret))
print('P-values ' + str(pval))
if pval < 0.05:    # alpha value is 0.05 or 5%
    print("we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
print('-----------------------------------------------------------------------------')

plot('2011','2019',12,12,df,dates_tempx,'ew',20,'ewretd')