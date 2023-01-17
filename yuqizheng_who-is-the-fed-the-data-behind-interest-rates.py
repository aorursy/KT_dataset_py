import pandas as pd
import datetime
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf 
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
%pylab inline 
pylab.rcParams['figure.figsize'] = (14, 8) # Change the size of plots
start = '1990-01-01' # Beginning of FOMC target rate data
end = '2018-11-14'
fomc = pd.read_csv('../input/fomc1.csv') # Load/format FOMC target data
fomc.set_index(pd.DatetimeIndex(fomc.Date), inplace=True)
fomc = fomc.resample('D', fill_method='pad').drop(['Date'], axis=1)
tbill13 = yf.download('^IRX', start, end) # Load tbill, bond data
ty5 = yf.download('^FVX', start, end)
bond10yr = yf.download('^TNX', start, end)
ty30 = yf.download('^TYX', start, end)
bonds = pd.DataFrame({'5 year': ty5['Adj Close'], # plot
                      '90 day': tbill13['Adj Close'],
                      '10 year': bond10yr['Adj Close'],
                      'FOMC rates': fomc['Level'],
                      '30 year': ty30['Adj Close']})
bonds.plot(grid = True)
plt.show()