# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import keras

from scipy.optimize import curve_fit

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
India_data = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv',parse_dates=['Date'], dayfirst=True)

India_data.drop(['Sno'], axis=1, inplace=True)

India_data.rename(columns={"State/UnionTerritory": "States"}, inplace=True)

India_data['Active Cases'] = India_data['Confirmed'] - India_data['Cured'] - India_data['Deaths']

India_daily= India_data.groupby(['Date'])['Active Cases'].sum().reset_index()
cases = India_data.groupby(['Date'])['Confirmed'].sum().values

daily = [cases[i] - cases[i-1] for i in range(1,cases.shape[0])]

plt.plot(cases)

plt.xlabel('No. of days')

plt.ylabel('No. of confirmed cases')

plt.title('Cummulative number of Corvid-19 cases in India')

fig = plt.gcf()

fig.set_size_inches(20, 10.5)
plt.plot(daily, 'g')

fig = plt.gcf()

plt.xlabel('No. of days')

plt.ylabel('No. of new cases')

plt.title('Daily new cases of Corvid-19 in India')

fig.set_size_inches(20, 10.5)
def sigmoid(x, a, b, c, d):

    x = [np.floor(-709*d + c) if -(z-c)/d > 709 else z for z in x] #to prevent math range error

    return [a + b/(1 + math.exp(-(z-c)/d)) for z in x]
plt.plot(sigmoid(np.linspace(-2000,2000,200), 0.01, 200000, 1000, 100));
# Current data

x = np.linspace(1, len(cases)+1, num=len(cases))

y = cases



# Last week's data

x_lw = x[:-7]

y_lw = y[:-7]



# Plot

plt.plot(x, y, label='Current')

plt.plot(x_lw, y_lw, label='Last week')

plt.xlabel('No. of days')

plt.ylabel('Confirmed cases')

plt.legend();

popt, pcov = curve_fit(sigmoid, x, y, bounds=((0, 0, 0, 0), (np.Inf, np.Inf, np.Inf, np.Inf))) # Fit current data to sigmoid

# Extent the curve to 210 days

length = 210

xt = np.linspace(1, length+1, num=length)

pred = sigmoid(xt, *popt)



popt_lw, pcov = curve_fit(sigmoid, x_lw, y_lw, bounds=((0, 0, 0, 0), (np.Inf, np.Inf, np.Inf, np.Inf))) # Fit last week's data to sigmoid

pred_lw = sigmoid(xt, *popt_lw) # Extent the curve to 210 days
print(popt)

print(popt_lw)
plt.plot(xt, pred, 'r', label='Fit based on current data')

plt.plot(xt, pred_lw, 'b', label="Fit based on last week's data")

plt.plot(cases, 'c', label='Actual data')

#plt.plot(xt, np.max(pred)*np.ones(xt.shape[0]), '^r')

plt.xlabel('No. of days')

plt.ylabel('No. of confirmed cases')

plt.title('Sigmoid fits for confirmed cases')

plt.legend()

fig = plt.gcf()

fig.set_size_inches(20, 10.5)
diff = [pred[i]-pred[i-1] for i in range(1, len(pred))]

diff_lw = [pred_lw[i]-pred_lw[i-1] for i in range(1, len(pred_lw))]

daily = [cases[i] - cases[i-1] for i in range(1,cases.shape[0])]

plt.plot(diff, 'r', label='Fit based on current data')

plt.plot(diff_lw, 'b', label="Fit based on last week's data")

plt.plot(daily[1:], 'g', label='Actual data')

plt.xlabel('No. of days')

plt.ylabel('No. of new cases')

plt.title('Fits for new cases')

plt.legend()

fig = plt.gcf()

fig.set_size_inches(20, 10.5)