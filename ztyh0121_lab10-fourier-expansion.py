import numpy as np
import pandas as pd
import time
import datetime as dt
import matplotlib.dates as md
import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))
coinbase=pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv')
coinbase.head(5)
print(coinbase.shape)
s = "01/01/2017"
sts=time.mktime(dt.datetime.strptime(s, "%d/%m/%Y").timetuple())
print(sts)
e = "01/01/2018"
ets=time.mktime(dt.datetime.strptime(e, "%d/%m/%Y").timetuple())
print(ets)
index=(coinbase['Timestamp']>=sts)&(coinbase['Timestamp']<ets)
coinbase17=coinbase[index]
coinbase17.shape
def totime(x):
    return dt.datetime.fromtimestamp(x).strftime('%m/%d %H:%M:%S')
coinbase17.index = list(map(totime,coinbase17['Timestamp'].values))
coinbase17.index
dates=[dt.datetime.fromtimestamp(ts) for ts in coinbase17['Timestamp'].values]
datenums=md.date2num(dates)
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax=plt.gca()
xfmt = md.DateFormatter('%m-%d')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(dates,coinbase17['Weighted_Price'].values)
plt.plot(dates,coinbase17['Volume_(BTC)'].values)
plt.show()
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax=plt.gca()
xfmt = md.DateFormatter('%m-%d')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(dates,coinbase17['Volume_(Currency)'].values/1000)
plt.show()
rdft=np.fft.rfft(coinbase17['Weighted_Price'])
print(coinbase17['Weighted_Price'].shape)
print(rdft.shape)
525600/2+1
dft=np.fft.fft(coinbase17['Weighted_Price'])
print(dft.shape)
freqs=np.fft.fftfreq(525600)
freqs
freqs2=freqs[1:]
print(freqs2.shape)
print(freqs2)
(525599+1)/2
freqs2[262799]
freqs2[262797:262802]
rfreqs=np.fft.rfftfreq(525600)
rfreqs
dft[0]
print(dft[1:262800].shape)
dft[1:262800]
print(dft[:262800:-1].shape)
dft[:262800:-1]
np.conj(dft[:262800:-1])
rdft.shape
rdft[0]
dft[0]
rdft[1:262800]
rdft[262800]
dft[262800]