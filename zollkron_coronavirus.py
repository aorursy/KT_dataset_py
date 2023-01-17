import pandas as pd
input_dir = "../input/spain-covid19-official-data-from-health-ministry/"

spain_data = pd.read_csv(input_dir + "coronavirus_dataset_spain.csv")

spain_data
spain_data.columns
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)
dates = list(spain_data['date'])

for i in range(len(dates)):

    dates[i] = dates[i].replace('2020-','')

dates
plt.bar(dates, spain_data['cases'])
spain_data['day'] = spain_data['day'].astype('float32')

spain_data['cases'] = spain_data['cases'].astype('float32')
spain_data.dtypes
dates

days = list(spain_data['day'])

cases = list(spain_data['cases'])

print(days, cases)
import numpy as np

X = np.array(days).reshape(-1, 1) 

y = np.array(cases).reshape(-1, 1)

X_test = np.array([12,13,14,15,16,17]).reshape(-1,1)
from scipy.optimize import curve_fit



def func(x, a, b, c):

    return a * (x * x) + b * x + c



popt, pcov = curve_fit(func, days, cases)

print(popt)



a = popt[0]

b = popt[1]

c = popt[2]



for i in range(12,17):

    y = func(i,a,b,c)

    print(y)
regions_data = pd.read_csv(input_dir + "coronavirus_dataset_spain_by_regions.csv")

regions_data
murcia_data = regions_data.loc[(regions_data['region'] == 'Murcia'),:]

murcia_data
for region in list(regions_data['region'].unique()):

    region_data = regions_data.loc[(regions_data['region'] == region),:]

    plt.figure()

    plt.rcParams["figure.figsize"] = (15,10)

    plt.rcParams['axes.titlesize'] = 20

    plt.ylabel('Casos')

    plt.xlabel('Fecha')

    plt.title(region)

    plt.plot(region_data['date'], region_data['cases'], label='Casos')

    dates = list(region_data['date'])

    cases = list(region_data['cases'])

    for i in range(len(dates)):

        #print(dates[i], cases[i])

        plt.text(dates[i], cases[i], cases[i])

    plt.plot(region_data['date'], region_data['deaths'], label='Muertos')

    deaths = list(region_data['deaths'])

    for i in range(len(dates)):

        #print(dates[i], deaths[i])

        plt.text(dates[i], deaths[i], deaths[i])

    plt.plot(region_data['date'], region_data['severes'], label='Graves (en la UCI)')

    severes = list(region_data['deaths'])

    for i in range(len(dates)):

        #print(dates[i], severes[i])

        plt.text(dates[i], severes[i], severes[i])

    plt.legend(loc='upper left')

    plt.show()