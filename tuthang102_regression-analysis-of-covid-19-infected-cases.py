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

from matplotlib.pylab import *
# Any results you write to the current directory are saved as output.
# data = pd.read_csv('covid_19_data.csv')
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
                         
rcParams.update({'font.size':21})
countries = data['Country/Region'].unique().tolist()
len(countries)
data['Date'] = data['ObservationDate'].apply(pd.to_datetime)
data.drop(['SNo'],axis=1,inplace=True)
data.tail()
data_new ={}
for name in countries:
    a = data[data['Country/Region']==name].groupby('Date').sum()
    data_new[name] = a.to_numpy()
figure(figsize=(10,7))
large_cases = []
for name in countries:
    confirm = data_new[name][:,0]
    if confirm[-1] > 2000:
        plot(confirm, 'o', label =name)
        large_cases.append(name)
legend(fontsize=10)
xlabel('Number of Days')
ylabel('Number of Cases')
grid('on')
show()

print('Number of cases with greater than 1000 confirmed cases:', len(large_cases))
large_cases
# A More Detailed Look at China Cases
from scipy.optimize import curve_fit

limit = 50
china_conf = data_new['Mainland China'][:,0][0:limit]
china_ded = data_new['Mainland China'][:,1][0:limit]
china_rev = data_new['Mainland China'][:,2][0:limit]
# Perform exponential fit for data 
def func(x, a, b, c, d):
    return a/(d + b*np.exp(-c * x))

xdata = np.arange(0,len(china_conf),1)
popt, pcov = curve_fit(func, xdata, china_conf)
figure(figsize=(8,5))
x_100 = np.arange(0,100,1)
plot(x_100, func(x_100, *popt), label='Model', linewidth=3)
plot(xdata, china_conf, 'o', label='Confirmed')
plot(xdata, china_rev, 'o', label='Recovered')
plot(xdata, china_ded, 'o', label='Death')
legend()
xlabel('Number of Days')
ylabel('Number of Cases')
grid('on')
show()
#Build a function to plot and fit any countries
def covid(name,days, start, stop):
    name_conf = data_new[name][:,0][start:stop]
    name_ded = data_new[name][:,1][start:stop]
    name_rev = data_new[name][:,2][start:stop]
    # Perform exponential fit for data 
    def func(x, a, b, c, d):
        return a/(d + b*np.exp(-c * x))

    xdata = np.arange(0,len(name_conf),1)
    x_100 = np.arange(0, days,1)
    popt, pcov = curve_fit(func, xdata, name_conf)
    figure(figsize=(7,4))
    plot(x_100, func(x_100, *popt), label='Model', linewidth=3)
    plot(xdata, name_conf, 'o', label='Confirmed')
    plot(xdata, name_rev, 'o', label='Recovered')
    plot(xdata, name_ded, 'o', label='Death')
    legend(fontsize=15)
    xlabel('Number of Days')
    ylabel('Number of Cases')
    title(name)
    grid('on')
    show()
covid('South Korea',80,20,80)
covid('Iran',60,0,50)
covid('Italy',60,25,60)
covid('Germany',70,25,70)
covid('US',70,25,70)
covid('Japan',100, 20, 100)
#Japan seems to be the best country in preventing the outbreak so far. 
figure(figsize=(8,5))
total_conf =[]
total_ded = []
total_rev = []
for name in large_cases:
    total_conf.append(data_new[name][-1,0])
    total_ded.append(data_new[name][-1,1])
    total_rev.append(data_new[name][-1,2])
total_conf = np.array(total_conf)
total_ded = np.array(total_ded)
total_rev = np.array(total_rev)
large_cases = np.array(large_cases)
figure(figsize=(30,10))
id = np.argsort(total_conf)

bar(np.arange(len(large_cases)), total_conf[id], align='center', alpha=0.3, label='Confirmed')
bar(np.arange(len(large_cases)), total_ded[id], align='center', alpha = 0.5, label = 'Death', color= 'red')
bar(np.arange(len(large_cases)), total_rev[id], align='center', alpha= 0.7, label='Recovered', color ='yellow')

xticks(np.arange(len(large_cases)), large_cases[id], fontsize=16)
xlabel('Country Name', fontsize=25)
ylabel('Total Number of Cases', fontsize=25)
title('Country with >2000 cases')
legend()
show()
tight_layout()
figure(figsize=(30,10))
ded_rate = total_ded/total_conf
idd = np.argsort(ded_rate)
bar(np.arange(len(large_cases)),ded_rate[idd], align='center')
xticks(np.arange(len(large_cases)), large_cases[idd], fontsize=16)
ylabel('Death rate', fontsize=25)
xlabel('Country Name', fontsize=25)
title('Death Rate')
show()
tight_layout()
figure(figsize=(20,10))
for i in range(len(ded_rate)):
    plot(total_conf[i], ded_rate[i], 'o', markersize=15)
    text(total_conf[i], ded_rate[i], large_cases[i], fontsize=15)
xlabel('Confirmed Cases')
ylabel('Death Rate')
title('Death Rate vs Confirmed Cases')
show()
tight_layout()

grow_rate = np.zeros(len(large_cases))
figure(figsize=(10,7))
i = 0
for name in large_cases:
    rate = np.gradient(data_new[name][:,0])
    rate = rate[rate>1]/np.max(data_new[name][:,0])
    if i%3 ==0:
        plot(rate, label=name)
    grow_rate[i] = np.mean(rate)
    i+=1
ylabel('Growth Rate')    
xlabel('Number of Days')
title('Growth Rate')
legend(fontsize=15)
idr = np.argsort(grow_rate)
figure(figsize=(30,10))
bar(np.arange(len(large_cases)),grow_rate[idr], align='center')
xticks(np.arange(len(large_cases)), large_cases[idr], fontsize=16)
ylabel('Growth rate', fontsize=25)
xlabel('Country Name', fontsize=25)
title('Average Growth Rate')
show()
tight_layout()
figure(figsize=(20,10))
for i in range(len(ded_rate)):
    plot(total_conf[i], grow_rate[i], 'o', markersize=15)
    text(total_conf[i], grow_rate[i], large_cases[i], fontsize=15)
xlabel('Confirmed Cases')
ylabel('Growth Rate')
title('Growth Rate vs Confirmed Cases')
show()
tight_layout()
