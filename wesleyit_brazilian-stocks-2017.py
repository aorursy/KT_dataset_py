import pandas_datareader as api
import matplotlib.pyplot as plt
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
start = datetime.datetime(2017,1,1)
end = datetime.datetime(2018,1,1)
stocks_prices = {}
my_stocks = ["BBAS3", "BBDC4", "BRML3", "BTOW3", "CRFB3", "GOLL4", 
             "ITUB4", "MOVI3", "NFLX34", "ORCL34", "PETR4", "SANB11", 
             "SEER3", "VALE3"]
# The regular API query via Pandas DataReader is not working on Kaggle
# due to Internect restrictions.
# I've solved the problem by creating an offline numpy dump of the dataset.

# Uncomment these lines to run on your computer:
# for stock in my_stocks:
#    stocks_prices[stock] = api.DataReader(stock, "google", start, end)

# Comment these lines to run on your computer:
stocks_prices = np.load('../input/offline_2017_stocks.np.npy').item()
for key in stocks_prices:
    print("Shape of ", key, ": ", len(stocks_prices[key]))
for stock in stocks_prices:
    plt.figure(figsize=[15,5])
    plt.title('Price for %s in 2017' % stock)
    plt.plot(stocks_prices[stock]['Close'])
    plt.grid()
    plt.xlabel("2017")
    plt.ylabel("R$")
    plt.show()
plt.figure(figsize=[18,6])
plt.plot(stocks_prices['BBAS3']['Close'],'y-', label="Brasil")
plt.plot(stocks_prices['BBDC4']['Close'],'r-', label="Bradesco")
plt.plot(stocks_prices['ITUB4']['Close'],'b-', label="Itau")
plt.plot(stocks_prices['SANB11']['Close'],'g-', label="Santander")
plt.grid()
plt.legend()
plt.xlabel("2017")
plt.ylabel("R$")
plt.show()
bbas = stocks_prices['BBAS3']
bbas_may = bbas[bbas.axes[0].month.isin(['05'])]
plt.figure(figsize=[18,6])
plt.plot(bbas_may['Volume'],'g-')
plt.grid()
plt.show()
plt.figure(figsize=[18,6])
plt.plot(bbas_may['Close'],'r-')
plt.grid()
plt.show()
bbas_may