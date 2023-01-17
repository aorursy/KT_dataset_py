import pandas as pd

import numpy as np



# plotting

import plotly.express as px



# for offline plotting

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)
def estimate(data, price, k):

    

    prices = data[price].to_numpy(dtype=np.float64)

    log_prices = np.log(prices)

    rets = np.diff(log_prices)

    T = len(rets)

    mu = np.mean(rets)

    var_1 = np.var(rets, ddof=1, dtype=np.float64)

    rets_k = (log_prices - np.roll(log_prices, k))[k:]

    m = k * (T - k + 1) * (1 - k / T)

    var_k = 1/m * np.sum(np.square(rets_k - k * mu))



    # Variance Ratio

    vr = var_k / var_1

    

    # Phi2

    def delta(j):

        res = 0

        for t in range(j+1, T+1):

            t -= 1  # array index is t-1 for t-th element

            res += np.square((rets[t]-mu)*(rets[t-j]-mu))

        return res / ((T-1) * var_1)**2



    phi2 = 0

    for j in range(1, k):

        phi2 += (2*(k-j)/k)**2 * delta(j)



    return vr, (vr - 1) / np.sqrt(phi2), T 
def estimate_multiple_k(data, price, time_intervals):

    

    # Estimate different time_intervals.

    for time_interval in time_intervals:



        vr, stat2, T = estimate(data, price, time_interval)

        print('The number of observations : ' + str(T))

        print('Variance Ratio for k = ' + str(time_interval) + ' : ' + str(vr))

        print('Variance Ratio Test Statistic for k = ' + str(time_interval) + ' Heteroscedasticity Assumption : ' + str(stat2))

        print('-------------------------------------------------------------------------------------------------')
us_index = pd.read_csv('../input/week1-data/week1_data.csv', na_values=['.'])

us_index['date'] = pd.to_datetime(us_index['Date'], format='%Y%m%d')

us_index.tail()
estimate_multiple_k(us_index, 'vindend', [2, 4, 8, 16,32])
def estimate_homo(data, price, k):

    

    prices = data[price].to_numpy(dtype=np.float64)

    log_prices = np.log(prices)

    rets = np.diff(log_prices)

    T = len(rets)

    mu = np.mean(rets)

    var_1 = np.var(rets, ddof=1, dtype=np.float64)

    rets_k = (log_prices - np.roll(log_prices, k))[k:]

    m = k * (T - k + 1) * (1 - k / T)

    var_k = 1/m * np.sum(np.square(rets_k - k * mu))



    # Variance Ratio

    vr = var_k / var_1

    

    # Phi_1

    phi1 = (2*(2*k-1)*(k-1))/(3*k*T)



    return vr, (vr - 1) / np.sqrt(phi1), T 
def estimate_multiple_k(data, price, time_intervals):

    

    # Estimate different time_intervals.

    for time_interval in time_intervals:



        vr, stat1, T = estimate_homo(data, price, time_interval)

        stat2 = estimate(data, price, time_interval)[1]

        print('The number of observations : ' + str(T))

        print('Variance Ratio for k = ' + str(time_interval) + ' : ' + str(vr))

        print('Variance Ratio Test Statistic for k = ' + str(time_interval) + ' Homoscedasticity Assumption : ' + str(stat1))

        print('Variance Ratio Test Statistic for k = ' + str(time_interval) + ' Heteroscedasticity Assumption : ' + str(stat2))

        print('-------------------------------------------------------------------------------------------------')
#goex etf involves in gold exploration

goex_etf = pd.read_csv('../input/ps1-dataset/GOEX (1).csv', na_values=['.'])

goex_etf['Date'] = pd.to_datetime(goex_etf['Date'], format='%d/%m/%Y')





#ARKW etf involves tech and internet

arkw_etf = pd.read_csv('../input/ps1-dataset/ARKW.csv', na_values=['.'])

arkw_etf['Date'] = pd.to_datetime(arkw_etf['Date'], format='%d/%m/%Y')





#SBIO etf invloves in healthcare 

sbio_etf = pd.read_csv('../input/ps1-dataset/SBIO.csv', na_values=['.'])

sbio_etf['Date'] = pd.to_datetime(sbio_etf['Date'], format='%d/%m/%Y')

import matplotlib.pyplot as plt

plt.scatter(goex_etf['Date'],goex_etf['Close'])

plt.title('GOEX')

plt.show()



plt.scatter(arkw_etf['Date'],arkw_etf['Close'])

plt.title('ARKW')

plt.show()



plt.scatter(sbio_etf['Date'],sbio_etf['Close'])

plt.title('SBIO')

plt.show()
time_intervals = [2,4,8,16]



print('GOEX','\n')

estimate_multiple_k(goex_etf,'Close',time_intervals)

print()

print('ARKW ETF','\n')

estimate_multiple_k(arkw_etf,'Close',time_intervals)

print()

print('SBIO ETF','\n')

estimate_multiple_k(sbio_etf,'Close',time_intervals)

print()