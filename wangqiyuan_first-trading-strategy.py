

# Imports

# pandas

import pandas as pd

from pandas import Series,DataFrame

import numpy as np



import datetime

import time



import matplotlib.finance as mpf

from matplotlib.pylab import date2num

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')

%matplotlib inline

spydata = pd.read_csv('../input/SPY Index.csv')

spydata.info()

spydata.head()
data_list = []

for index, row  in spydata.iterrows():

    date_time = datetime.datetime.strptime(row['Date'],'%m/%d/%Y')

    t = date2num(date_time)

    open,high,low,close = row['Open'], row['High'],row['Low'], row['Close']

    datas = (t,open,high,low,close)

    data_list.append(datas)

    

fig, ax = plt.subplots()

fig.set_size_inches(30, 15)

fig.subplots_adjust(bottom=0.4)

ax.xaxis_date()

plt.xticks(rotation=45)

plt.yticks()

plt.title("SPY 1993-2017")

plt.xlabel("time")

plt.ylabel("price")

mpf.candlestick_ohlc(ax,data_list,width=1.5,colorup='r',colordown='green')

plt.grid() 
data_list = []

for item in spydata['Date']:    

    date_time = datetime.datetime.strptime(item,'%m/%d/%Y')

    data_list.append(date_time)

spydata['Date'] = data_list

fig = plt.gcf()

fig.set_size_inches(30, 15)

plt.plot(spydata['Date'], spydata['Close'])
spydata['up or down'] = 0

spydata['upper bound'] = 0.0

spydata['lower bound'] = 0.0

a = spydata.shape

length = a[0]-1

for i in range(1,length):

    if ((spydata['High'].iat[i]-spydata['High'].iat[i-1] > 0) & (spydata['High'].iat[i]-spydata['High'].iat[i+1] > 0) & (spydata['Low'].iat[i]-spydata['Low'].iat[i-1] > 0) & (spydata['Low'].iat[i]-spydata['Low'].iat[i+1] > 0)):

        spydata['up or down'].iat[i] = 1

    if ((spydata['High'].iat[i]-spydata['High'].iat[i-1] < 0) & (spydata['High'].iat[i]-spydata['High'].iat[i+1] < 0) & (spydata['Low'].iat[i]-spydata['Low'].iat[i-1] < 0) & (spydata['Low'].iat[i]-spydata['Low'].iat[i+1] < 0)):

        spydata['up or down'].iat[i] = -1

    

for i in range(3,length):

    if(spydata['up or down'].iat[i-2] == 1):

        spydata['upper bound'].iat[i] = spydata['High'].iat[i-2]

    else:

        spydata['upper bound'].iat[i] = spydata['upper bound'].iat[i-1]

        

for i in range(3,length):

    if(spydata['up or down'].iat[i-2] == -1):

        spydata['lower bound'].iat[i] = spydata['Low'].iat[i-2]

    else:

        spydata['lower bound'].iat[i] = spydata['lower bound'].iat[i-1]



#spydata['lower bound']

#spydata['upper bound']    

#spydata['up or down']

spydata.head()

#spydata['up or down'].value_counts()
spydata['upper signal'] = 0.0

spydata['lower signal'] = 0.0

num = 15

for i in range(num,a[0]-num):

    max_value = 0.0

    min_value = 1000.0

    for j in range(num+1):

        max_value = max(max_value,spydata['upper bound'].iat[i-j])

        min_value = min(min_value,spydata['lower bound'].iat[i-j])

    spydata['upper signal'].iat[i] = max_value

    spydata['lower signal'].iat[i] = min_value

    

fig = plt.gcf()

fig.set_size_inches(30, 15)

plt.plot(spydata['Date'], spydata['lower signal'])

plt.plot(spydata['Date'], spydata['Close'])

plt.plot(spydata['Date'], spydata['upper signal'])

plt.show()



spydata.head()

#spydata.index = spydata['Date']

spydata[['lower signal','Close','upper signal']].ix[50:1500].plot(figsize=(12,7))

       
spydata['returns'] = np.log(spydata['Close']/spydata['Close'].shift(1))

plt.plot(spydata['Date'], spydata['returns'])

spydata.head()
spydata['rolling_mean5']=spydata['Close'].rolling(window = 5).mean()

spydata['rolling_mean20']=spydata['Close'].rolling(window = 20).mean()

spydata['rolling_High5']=spydata['High'].rolling(window = 5).mean()

spydata['rolling_High20']=spydata['High'].rolling(window = 20).mean()

spydata['rolling_Low5']=spydata['Low'].rolling(window = 5).mean()

spydata['rolling_Low20']=spydata['Low'].rolling(window = 20).mean()

spydata['rolling_volume']=spydata['Volume'].rolling(window = 5).mean()

spydata['rolling_std_5']=spydata['Open'].rolling(window = 5).std()

spydata['rolling_std_20']=spydata['Open'].rolling(window = 20).std()

#spydata['rolling_max']=spydata['High'].rolling(window = 10).max()

print(np.nanmean(spydata['rolling_std_5']))

print(np.nanmean(spydata['rolling_mean5']-spydata['rolling_mean20']))

print(np.nanstd(spydata['rolling_mean5']-spydata['rolling_mean20']))





fig = plt.gcf()

fig.set_size_inches(30, 15)

ax1 = fig.add_subplot(111)

ax1.plot(spydata['Date'], spydata['rolling_volume'])

ax1.set_ylabel('SPY rolling_volume')

ax2 = ax1.twinx() 

ax2.plot(spydata['Date'], spydata['rolling_std_5'], 'r')

ax2.set_ylabel('rolling_std')

plt.show()
fig = plt.gcf()

fig.set_size_inches(30, 15)

ax1 = fig.add_subplot(111)

ax1.plot(spydata['Date'], spydata['Close'])

ax1.set_ylabel('SPY_Close')

ax2 = ax1.twinx() 

ax2.plot(spydata['Date'], spydata['rolling_std_5'], 'r')

ax2.set_ylabel('rolling_std_5')

plt.show()
spydata[['rolling_mean5','Close','rolling_mean20']].ix[3500:4500].plot(figsize=(12,7))
spydata['buyorsell_2'] = 0

for i in range(num,length):

    if(spydata['rolling_mean5'].iat[i]-spydata['rolling_mean20'].iat[i] > 0.24) & (spydata['High'].iat[i] < spydata['rolling_High5'].iat[i]):

        spydata['buyorsell_2'].iat[i] = 1

    if(spydata['rolling_mean20'].iat[i]-spydata['rolling_mean5'].iat[i] > 0.24) & (spydata['Low'].iat[i] > spydata['rolling_Low5'].iat[i]):

        spydata['buyorsell_2'].iat[i] = -1

spydata['buyorsell_2'].value_counts()
fig = plt.figure()

fig.set_size_inches(30, 15)

ax1 = fig.add_subplot(111)

ax1.plot(spydata['Date'], spydata['Close'])

ax1.set_ylabel('SPY Close Price')



ax2 = ax1.twinx() 

ax2.plot(spydata['Date'], spydata['buyorsell_2'], 'r')

ax2.set_ylabel('signal')

plt.show()
spydata['buyorsell'] = 0

spydata['closeposition'] = 0.0

for i in range(num,length):

    spydata['closeposition'].iat[i] = (spydata['High'].iat[i]+spydata['Low'].iat[i])/2

    #if ((spydata['Close'].iat[i]-spydata['upper signal'].iat[i] > 0) & ((spydata['High'].iat[i]-spydata['Low'].iat[i])/spydata['Open'].iat[i] > 0.01) ):

    #if ((spydata['Close'].iat[i]-spydata['upper signal'].iat[i] > 0) & (spydata['Volume'].iat[i]/spydata['Volume'].iat[i-1] > 1.5) ):

    if((spydata['Close'].iat[i]-spydata['upper signal'].iat[i] > 0)& (spydata['Volume'].iat[i] > spydata['rolling_volume'].iat[i])):

        spydata['buyorsell'].iat[i] = 1

    #if ((spydata['Close'].iat[i]-spydata['lower signal'].iat[i] < 0) & ((spydata['High'].iat[i]-spydata['Low'].iat[i])/spydata['Open'].iat[i] > 0.01) ) : 

    #if ((spydata['Close'].iat[i]-spydata['lower signal'].iat[i] < 0) & (spydata['Volume'].iat[i]/spydata['Volume'].iat[i-1] > 1.5) ):

    if((spydata['Close'].iat[i]-spydata['lower signal'].iat[i] < 0) & (spydata['Volume'].iat[i] > spydata['rolling_volume'].iat[i]) ):

        spydata['buyorsell'].iat[i] = -1

        

#fig = plt.gcf()

#fig.set_size_inches(30, 15)

#plt.plot(spydata['Date'], spydata['closeposition'])

spydata['buyorsell'].value_counts()
fig = plt.figure()

fig.set_size_inches(30, 15)

ax1 = fig.add_subplot(111)

ax1.plot(spydata['Date'], spydata['Close'])

ax1.set_ylabel('SPY Close Price')



ax2 = ax1.twinx() 

ax2.plot(spydata['Date'], spydata['buyorsell'], 'r')

ax2.set_ylabel('signal')

plt.show()
spydata['buyorsell_1'] = 0

for i in range(num,length):

    if((spydata['Close'].iat[i]-spydata['upper signal'].iat[i] > 0)& (spydata['rolling_std_5'].iat[i] > spydata['rolling_std_20'].iat[i])):

        spydata['buyorsell_1'].iat[i] = 1

    if((spydata['Close'].iat[i]-spydata['lower signal'].iat[i] < 0) & (spydata['rolling_std_5'].iat[i] > spydata['rolling_std_20'].iat[i]) ):

        spydata['buyorsell_1'].iat[i] = -1

spydata['buyorsell_1'].value_counts()
fig = plt.figure()

fig.set_size_inches(30, 15)

ax1 = fig.add_subplot(111)

ax1.plot(spydata['Date'], spydata['Close'])

ax1.set_ylabel('SPY Close Price')



ax2 = ax1.twinx() 

ax2.plot(spydata['Date'], spydata['buyorsell_1'], 'r')

ax2.set_ylabel('signal')

plt.show()
#trend trading Today's open > yesterday close, Today's close near today's high,  Today's close > 10_max high

spydata['buy_signal']= 0 

spydata['10_max'] = 0

#spydata['10_min'] = 0

num_max = 10

for i in range(num_max,a[0]-num_max):

    max_value = 0.0

    #min_value = 1000.0

    for j in range(num_max+1):

        max_value = max(max_value,spydata['High'].iat[i-j])

        #min_value = min(min_value,spydata['Low'].iat[i-j])

    spydata['10_max'].iat[i] = max_value

    #spydata['10_min'].iat[i] = min_value



print(spydata['10_max'].max())    

#print(spydata['10_min'].max()) 

    

for i in range(num_max,length):

    #if((spydata['Open'].iat[i]-spydata['Close'].iat[i-1] > 0) & (spydata['Close'].iat[i] > spydata['10_max'].iat[i]) & (spydata['Close'].iat[i] > 0.99*spydata['High'].iat[i])):

    if((spydata['Open'].iat[i]-spydata['Close'].iat[i-1] > 0) & (spydata['Close'].iat[i] > spydata['10_max'].iat[i]) & (spydata['Close'].iat[i] > 0.99*spydata['High'].iat[i])):    

        spydata['buy_signal'].iat[i] = 1

    #if((spydata['Open'].iat[i]-spydata['Close'].iat[i-1] < 0) & (spydata['Close'].iat[i] < spydata['10_min'].iat[i]) & (spydata['Close'].iat[i] < 0.99*spydata['Low'].iat[i])):

    #if((spydata['Open'].iat[i]-spydata['Close'].iat[i-1] < 0) & (spydata['Low'].iat[i] <= spydata['10_min'].iat[i])):    

        #spydata['buy_signal'].iat[i] = -1    



spydata['buy_signal'].value_counts()
ax1 = plt.subplot(311)

plt.plot(spydata['Date'], spydata['Close'])

plt.setp(ax1.get_xticklabels(), fontsize=6)

ax2 = plt.subplot(312, sharex=ax1)

plt.plot(spydata['Date'], spydata['buy_signal'])
'''

spydata['signal'] = 0

spydata['buyorsell_1'] = spydata['buyorsell']

i = num

while i<= length:

    AF = 0.0

    high = spydata['High'].iat[i]

    low = spydata['Low'].iat[i]

    while(spydata['buyorsell_1'].iat[i] == 1):

        spydata['signal'].iat[i] = 1

        #spydata['closeposition'].iat[i] = spydata['lower siganl'].iat[i-1]

        i +=1

        if(spydata['High'].iat[i]>high):

            high = spydata['High'].iat[i]

            AF = AF + 0.01

            if (AF >= 0.1):

                AF = 0.1

        spydata['closeposition'].iat[i] = spydata['closeposition'].iat[i-1] +(spydata['upper signal'].iat[i]- spydata['closeposition'].iat[i-1])*AF

        if(spydata['closeposition'].iat[i] > spydata['Close'].iat[i] ):

            spydata['signal'].iat[i] = 1

            spydata['buyorsell_1'].iat[i] = 1

        else:

            break

                      

spydata['signal'].value_counts()



'''        



        
AF = 0.01

for i in range(num,length):

    if(spydata['buyorsell'].iat[i] == 1):

        spydata['closeposition'].iat[i] = spydata['closeposition'].iat[i-1] + (spydata['upper signal'].iat[i]- spydata['closeposition'].iat[i-1])*AF        

    if(spydata['buyorsell'].iat[i] == -1):

        spydata['closeposition'].iat[i] = spydata['closeposition'].iat[i-1] - (spydata['closeposition'].iat[i-1]-spydata['lower signal'].iat[i])*AF        
spydata['signal'] = 0

for i in range(num,length):

    if ((spydata['buyorsell'].iat[i] == 1) & (spydata['closeposition'].iat[i]-spydata['Open'].iat[i] > 0)):

        spydata['signal'].iat[i] = 1

    if ((spydata['buyorsell'].iat[i] == -1) & (spydata['closeposition'].iat[i]-spydata['Open'].iat[i] < 0)):

        spydata['signal'].iat[i] = -1

        

spydata['signal'].value_counts()

        
'''

fig = plt.figure()

fig.set_size_inches(30, 15)

ax1 = fig.add_subplot(111)

ax1.plot(spydata['Date'], spydata['Close'])

ax1.set_ylabel('SPY Close Price')



ax2 = ax1.twinx() 

ax2.plot(spydata['Date'], spydata['buyorsell'], 'r')

ax2.set_ylabel('signal')

plt.show()



spydata[['Close','buyorsell']].ix[50:1500].plot(figsize=(12,7))

'''
ax1 = plt.subplot(311)

plt.plot(spydata['Date'], spydata['Close'])

plt.setp(ax1.get_xticklabels(), fontsize=6)

ax2 = plt.subplot(312, sharex=ax1)

plt.plot(spydata['Date'], spydata['buyorsell'])
spy_signals = pd.concat([

    pd.DataFrame({

        "Date": spydata.loc[spydata['buyorsell_2'] == 1,"Date"],

        "Price": spydata.loc[spydata['buyorsell_2'] == 1,"Close"],

        "Regime": spydata.loc[spydata['buyorsell_2'] == 1,"buyorsell_2"],

        "Signal": "Buy" }),

    pd.DataFrame({

         "Date": spydata.loc[spydata['buyorsell_2'] == -1,"Date"],

        "Price": spydata.loc[spydata['buyorsell_2'] == -1,"Close"],

        "Regime": spydata.loc[spydata['buyorsell_2'] == -1,"buyorsell_2"],

        "Signal": "Sell" }),

])

spy_signals.sort_index(inplace = True)

spy_signals



#spydata['stragety'] = spydata.buyorsell.shift(1) * spydata.returns

#spydata[['stragety','returns']].cumsum().apply(np.exp).plot(figsize=(12,7))

  
#trend trading, spydata['Volume'].iat[i] > spydata['rolling_volume'].iat[i]

spydata = spydata.set_index('Date')

spydata['stragety'] = spydata.buyorsell.shift(1) * spydata.returns

spydata[['stragety','returns']].cumsum().apply(np.exp).plot(figsize=(12,7))
#trend trading,spydata['rolling_std_5'].iat[i] > spydata['rolling_std_20']

spydata['stragety0'] = spydata.buyorsell_1.shift(1) * spydata.returns

spydata[['stragety0','returns']].cumsum().apply(np.exp).plot(figsize=(12,7))
#rolling_mean5,20--high, rolling_high, low, rolling_low

spydata['stragety4'] = spydata.buyorsell_2.shift(1) * spydata.returns

spydata[['stragety4','returns']].cumsum().apply(np.exp).plot(figsize=(12,7))
#SAR

spydata['stragety1'] = spydata.signal.shift(1) * spydata.returns

spydata[['stragety1','returns']].cumsum().apply(np.exp).plot(figsize=(12,7))
#trend trading Today's open > yesterday close, Today's close near today's high,  Today's close > 10_max high

spydata['stragety2'] = spydata.buy_signal.shift(1) * spydata.returns

spydata[['stragety2','returns']].cumsum().apply(np.exp).plot(figsize=(12,7))