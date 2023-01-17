# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
eu_odp_df = pd.read_excel('https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx')
ru_cases_df = eu_odp_df[eu_odp_df['countryterritoryCode'] == 'RUS']
ru_cases_df.dateRep.max()
%matplotlib inline
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
begging_pred = ru_cases_df['dateRep'] > '2020-03-11'
ru_cases_df = ru_cases_df[begging_pred]
ru_cases_df['dateRepInt'] = ru_cases_df['dateRep'].astype(int)
from scipy.optimize import curve_fit
def func(t, a, b, c, d):

#     return a*np.exp(b*t) + c

    return a*t**2+b*t+c
dt = ru_cases_df['dateRepInt'].values

dt
cases = ru_cases_df['cases'].values

cases
params_guess = np.array([cases.min(),cases.max(),cases.mean(),cases.std()])
popt, pcov = curve_fit(func, dt, cases, p0=params_guess, maxfev=1000)
p1 = np.polyfit(dt,cases,3)

fit_eq = np.poly1d(p1)

fit_eq
ru_cases_df['curve_fit'] = func(dt, *popt)
n=len(cases)



err = np.sqrt(np.sum((pow((np.array(ru_cases_df['curve_fit'].values)-cases),2)))/n)

print(err)



tss = np.sum(pow((cases - np.mean(cases)),2))

rss = np.sum(pow((cases - np.array(ru_cases_df['curve_fit'].values)),2))



r2 = 100*(1-(rss/tss))

print(r2)



sd = pow((rss/(n-2)),0.5)

print(sd)
ru_cases_df['poly_fit'] = fit_eq(dt)
n=len(cases)



err = np.sqrt(np.sum((pow((np.array(ru_cases_df['poly_fit'].values)-cases),2)))/n)

print(err)



tss = np.sum(pow((cases - np.mean(cases)),2))

rss = np.sum(pow((cases - np.array(ru_cases_df['poly_fit'].values)),2))



r2 = 100*(1-(rss/tss))

print(r2)



sd = pow((rss/(n-2)),0.5)

print(sd)
ax = ru_cases_df.plot(x='dateRepInt',y='cases',color='orange')

ru_cases_df.plot(ax=ax,x='dateRepInt',y='deaths',color='red')

ru_cases_df.plot(ax=ax,x='dateRepInt',y='poly_fit',color='blue')

ax.set_title('Timeseries')

ax.set_xlabel('Date report')

ax.autoscale(enable=True)
new_dt = np.arange(np.min(dt), np.max(dt)+14*(dt[0]-dt[1]), dt[0]-dt[1])
import matplotlib.ticker as plticker



loc = plticker.MultipleLocator(base=dt[0]-dt[1]) # this locator puts ticks at regular intervals
new_y = fit_eq(new_dt)

fig, ax = plt.subplots()

ax.plot(new_dt, new_y)

ax.plot(dt, cases,color='red')

ax.set(xlabel='days', ylabel='cases per day')

ax.xaxis.set_major_locator(loc)

ax.grid()

plt.savefig('foo.png')

plt.show()
new_y = np.cumsum(np.sort(fit_eq(new_dt)))

_, ax1 = plt.subplots()

ax1.plot(new_dt, new_y)

ax1.grid()

ax1.plot(dt, np.flip(np.cumsum(np.sort(cases))), color='red')

ax1.set(xlabel='days', ylabel='cases total')

ax1.xaxis.set_major_locator(loc)

plt.show()