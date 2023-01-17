### Imports ###

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import altair as alt # Interactive data visualization

import os

from datetime import date, timedelta

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



# Listing all files available

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading deaths and confirmed cases to memory

deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
# Summing up the regions of countries

df = confirmed.groupby('Country/Region').sum()
def plotGR(country, days = None, start = None):

    '''Function to plot and return exponential growth rate µ.

    Exponential growth: N(t) = N0 * exp(µ * t)

    At µ = 0.693 the population is doubling once per day'''

    

    valid = df.loc[country,'1/22/20':][df.loc[country,'1/22/20':] > 100].values[start:days]

    f = np.polyfit(np.arange(len(valid)), np.log(valid), 1)

    f = np.poly1d(f)

    t = np.arange(len(valid))



    plt.scatter(np.arange(len(valid)), np.log(valid), label = f'{country}, µ = {round(f[1], 2)}, tD = {round(np.log(2)/f[1],1)} days')

    plt.plot(t, f(t), label = '_nolabel')



    plt.legend(bbox_to_anchor = (1,1))

    return(f)
# Countries with signifficant number of confirmed cases

high = list(df[(df.iloc[:,-14] > 100) &

               (df.iloc[:,-1] > 5000)].index)
# Plotting Growth rates

growth_rates = []

plt.figure(dpi = 144)

for country in high:

    growth_rates.append(plotGR(country, 7))

plt.xlabel('Days after reaching 100 confirmed infections')

plt.ylabel('ln(confirmed cases)')

plt.title(f'''Average Initial growth rate µ = {round(np.mean([a[1] for a in growth_rates]),3)} 

+/- {round(np.std([a[1] for a in growth_rates]),3)} (std. dev.)''')
# Summing up the regions of countries

df2 = deaths.groupby('Country/Region').sum()

# Focusing on countries with at least 100 confirmed deaths (as of 26.3.2020)

high_deaths = df2.loc[df2.loc[:,'3/26/20'] > 100,:].index
def plotGR_death(country, days = None, start = None):

    '''Function to plot and return exponential growth rate µ.

    Exponential growth: N(t) = N0 * exp(µ * t)

    At µ = 0.693 the population is doubling once per day'''

    

    valid = df2.loc[country,'1/22/20':][df2.loc[country,'1/22/20':] > 10].values[start:days]

    f = np.polyfit(np.arange(len(valid)), np.log(valid), 1)

    f = np.poly1d(f)

    t = np.arange(len(valid))



    plt.scatter(np.arange(len(valid)), np.log(valid), label = f'{country}, µ = {round(f[1], 2)}, tD = {round(np.log(2)/f[1],1)} days')

    plt.plot(t, f(t), label = '_nolabel')



    plt.legend(bbox_to_anchor = (1,1))

    return(f)
## Same as above, just based on deceased

growth_rates_death = []

plt.figure(dpi = 144)

for country in high_deaths:

    growth_rates_death.append(plotGR_death(country, 7))

l = plt.xlabel('Days after reaching 10 confirmed deaths')

l = plt.ylabel('ln(deaths)')

l = plt.title(f'''Average Inital growth rate (based on deaths) µ = {round(np.mean([a[1] for a in growth_rates_death]),3)} 

+/- {round(np.std([a[1] for a in growth_rates_death]),3)} (std. dev.)''')
plt.figure(dpi = 144, figsize = (6,5))

for i, country in enumerate(high):

    if i < 9:

        p1 = plt.subplot(2,1,1)

        plotGR(country, start = -7)

    else:

        p2 = plt.subplot(2,1,2, sharex = p1)

        plotGR(country, start = -7)



l = plt.ylabel('ln(confiremd cases)')

l = plt.suptitle('Growth rate of confirmed cases during the last week')
plt.figure(dpi = 144, figsize = (6,5))

for i, country in enumerate(high):

    if i < 9:

        p1 = plt.subplot(2,1,1)

        plotGR_death(country, start = -7)

    else:

        p2 = plt.subplot(2,1,2, sharex = p1)

        plotGR_death(country, start = -7)



l = plt.ylabel('ln(deaths)')

l = plt.suptitle('Growth rate of deaths during the last week')
# Sliding window fit 

def slidingWindowFit(df, lower_cases_limit = 100):

    window_len = 7 # 7 days sliding window

    growth_rates = pd.DataFrame([])

    for country in df.index:

        ts = df.loc[country,'1/22/20':]

        mu = {}

        for start in range(3, len(ts) - 3):

            if ts[start] > lower_cases_limit: # We are above the noise threshold

                f = np.polyfit(np.arange(window_len), np.log(ts[start - 3:start + 4]), 1)

                f = np.poly1d(f)

                mu[ts.index[start]] = f[1]

            else:

                mu[ts.index[start]] = np.nan

        

        growth_rates = pd.concat([growth_rates, pd.DataFrame(mu, index = [country])])

    return (growth_rates)
growth_rates = slidingWindowFit(df)

growth_rates_death = slidingWindowFit(df2, 10)
plt.figure(dpi = 144)

for country in high:

    plt.plot( growth_rates.loc[country,:], label = country)

l = plt.xticks(range(0,len(growth_rates.columns),2),growth_rates.columns[0:-1:2], fontSize = 6, rotation = 60)

l = plt.legend(bbox_to_anchor = (1,1))

l = plt.xlabel('Date')

l = plt.ylabel('7 day sliding window exponential growth')

l = plt.title('Growth Rates over time (7 day sliding window fit)')
plt.figure(dpi = 144)

for country in high:

    plt.plot( growth_rates_death.loc[country,:], label = country)

l = plt.xticks(range(0,len(growth_rates.columns),2),growth_rates.columns[0:-1:2], 

               fontSize = 6, rotation = 60)

l = plt.legend(bbox_to_anchor = (1,1))

l = plt.xlabel('Date')

l = plt.ylabel('7 day sliding window exponential growth')

l = plt.title('Growth Rates of deaths over time (7 day sliding window fit)')
td = np.log(2) / growth_rates.iloc[:,-1].dropna()

td_death = np.log(2) / growth_rates_death.iloc[:,-1]

print('Doubling times of confirmed cases | deaths in days (tD)')

print('Date:',str(date.today()))

print('------------')

l = [print(a,':',round(b,1),'|', round(td_death.loc[a],1)) for (a, b) in zip(td.index, td)]
for country in high:

    plt.figure()

    plt.plot( growth_rates_death.loc[country,:], label = 'Deaths')

    plt.plot(growth_rates.columns, growth_rates.loc[country,:], label = 'Confirmed cases')

    l = plt.xticks(range(0,len(growth_rates.columns),2),growth_rates.columns[0:-1:2], 

                   fontSize = 6, rotation = 60)

    l = plt.legend(bbox_to_anchor = (1,1))

    l = plt.xlabel('Date')

    l = plt.ylabel('7 day sliding window exponential growth rate')

    l = plt.title(f'Growth Rates of COVID-19 in {country}')

    plt.show()
from mpl_toolkits.basemap import Basemap
gr = pd.merge(df.iloc[:,:2],growth_rates, left_index = True, right_index = True)

grd = pd.merge(df.iloc[:,:2],growth_rates_death, left_index = True, right_index = True)
plt.figure(dpi = 300)

m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)

m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

m.fillcontinents(color='grey', alpha = 0.3)

m.drawcoastlines(linewidth=0.1, color="white")

m.scatter(df['Long'], df['Lat'], latlon = True, c = np.log(df.iloc[:,-1]), s = 10, alpha = 0.8, cmap='jet')

#m.contourf(df['Long'].values, df['Lat'].values,  data = np.log(df.iloc[:,-1].fillna(0.001).values))#, latlon = True)

m.colorbar()

l = plt.title('Natural Logarithm of Confirmed Cases')
plt.figure(dpi = 300)

m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)

m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

m.fillcontinents(color='grey', alpha = 0.3)

m.drawcoastlines(linewidth=0.1, color="white")

m.scatter(df2['Long'], df2['Lat'], latlon = True, c = np.log(df2.iloc[:,-1]), s = 10, alpha = 0.8, cmap='jet')

#m.contourf(df['Long'].values, df['Lat'].values,  data = np.log(df.iloc[:,-1].fillna(0.001).values))#, latlon = True)

m.colorbar()

l = plt.title('Natural Logarithm of Deaths')
plt.figure(dpi = 300)

m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)

m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

m.fillcontinents(color='grey', alpha = 0.3)

m.drawcoastlines(linewidth=0.1, color="white")

m.scatter(gr['Long'], gr['Lat'], latlon = True, c = gr.iloc[:,-1], s = 5, alpha = 0.8, cmap='jet', vmax = 0.05)

#m.contourf(df['Long'].values, df['Lat'].values,  data = np.log(df.iloc[:,-1].fillna(0.001).values))#, latlon = True)

m.colorbar()

l = plt.title('Growth Rate of Confirmed Cases \n(> 0.05 = Virus spreading exponentially)')
plt.figure(dpi = 300)

m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)

m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

m.fillcontinents(color='grey', alpha = 0.3)

m.drawcoastlines(linewidth=0.1, color="white")

m.scatter(grd['Long'], grd['Lat'], latlon = True, c = grd.iloc[:,-1], s = 10, alpha = 0.8, cmap='jet')

#m.contourf(df['Long'].values, df['Lat'].values,  data = np.log(df.iloc[:,-1].fillna(0.001).values))#, latlon = True)

m.colorbar()

l = plt.title('Growth Rate of Deaths')
def gompertz(t, N0, K, b):

    Nt = K * np.exp(np.log(N0 / K) * np.exp(- b * t))

    return (Nt)
from scipy.optimize import curve_fit



y_data = df2.loc[df2.index != 'China','2/22/20':].sum()

#y_data = df2.loc['US','2/22/20':]

y_data  = y_data[y_data > 0]

x_data = np.arange(len(y_data)) + 1





popt, pcov = curve_fit(gompertz,x_data, y_data, bounds = (

    [1, 1, 0.001], [1000, 1e10 , 1] ))
def plotGompertz(country = 'Global (excl. China)', fig = None, deceased = True):

    if deceased:

        dfp = df2

        name = 'Deaths'

    else:

        dfp = df

        name = 'Reported Cases'

        

    if country == 'Global (excl. China)':

        y_data = dfp.loc[dfp.index != 'China','2/22/20':].sum()

    else:

        y_data = dfp.loc[country,'2/22/20':]

        

    y_data  = y_data[y_data > 0]

    x_data = np.arange(len(y_data)) + 1

    

    time = pd.to_datetime(y_data.index)

    time_extra = pd.date_range(time[0], time[-1] + timedelta(days = 90))



    popt, pcov = curve_fit(gompertz,x_data, y_data, bounds = (

        [0, 1e2, 0.001], [y_data[0] * 2, 1e9 , 1] ))



    print(popt)

    if not fig:

        fig = plt.figure(dpi = 144, figsize = (8, 8))

    p1 = plt.subplot(311)

    plt.scatter(time, y_data, label = f'{name} {country}')

    plt.scatter(time, gompertz(x_data , *popt), s = 5, label = f'Model fit {name} {country}')

    plt.ylabel('Total #')

    plt.title(f'Gompertz model fit')

    plt.legend(bbox_to_anchor = (1,1))





    plt.subplot(312, sharex=p1)

    plt.plot(time, gompertz(np.arange(len(time)), *popt), label = f'{name} {country}', linewidth='2')

    plt.scatter(time_extra, gompertz(np.arange(len(time_extra)), *popt), label = '_nolabel',  s = 1)

    plt.ylabel('Total')

    plt.title('Extrapolation of Model')

    plt.legend(bbox_to_anchor = (1,1))



    plt.subplot(313, sharex=p1)

    plt.plot(time, np.diff(gompertz(np.arange(len(time) + 1) , *popt)),label = f'{name} {country}' , linewidth='2')

    plt.scatter(time_extra, np.diff(gompertz(np.arange(len(time_extra) + 1) , *popt)), label = f'_nolabel', s = 1)

    

    p_max = np.max(np.diff(gompertz(np.arange(len(time_extra) + 1) , *popt)))

    plt.plot([date.today(),date.today()], [0, p_max], color='red')

    plt.ylabel('Daily increase in #')

    plt.title('Extrapolation of Model - Daily increase')

    plt.tight_layout()

    plt.legend(bbox_to_anchor = (1,1))

    

    return(fig)
f = plotGompertz()
f = plotGompertz('Italy', deceased = False)

f = plotGompertz('Spain', f, deceased = False)

f = plotGompertz('Germany', f, deceased = False)

f = plotGompertz('France', f, deceased = False)
f = plotGompertz('Italy', deceased = True)

f = plotGompertz('Spain', f, deceased = True)

f = plotGompertz('Germany', f, deceased = True)

f = plotGompertz('France', f, deceased = True)
f = plotGompertz('Italy',  deceased = False)

f = plotGompertz('Spain', f, deceased = False)

f = plotGompertz('US', f, deceased = False)
# Exponential decay of growth rate

def mu_with_decay(t, µ0, lam, b):

    µ = µ0  * (np.exp(lam * t) + b)

    return(µ)
from scipy.integrate import odeint



def SIR(y, t, µ_params):

    # Extract data from vector

    S, I, R = y 

    

    # From Growth rate decay model

    µ0, lam, b = µ_params

    µ = µ0  * (np.exp(lam * t) + b)

    

    ### CONSTANTS ###

    #c = 0.26 # Growth constant

    #p = 0.37 # Policy factor, to take into account things lice social distancing, shutdowns etc. (range=(0,1))

    omega = 1 / 30 # Recovery rate

    N = S + I + R # Total Population size

    

    ### ODE system ###

    dSdt = - µ * S / N * I 

    dIdt =   µ * S / N * I - omega * I

    dRdt = omega * I

    

    return([dSdt, dIdt, dRdt])
# First,  getting growth rate parameters



for country in ['China','US','Austria','Denmark','Germany']:

    y_data = growth_rates.loc[country].dropna().values[:]

    x_data = np.arange(len(y_data))



    popt, pcov = curve_fit(mu_with_decay, x_data, y_data, bounds = ([0.1, -1, 0],[1, 0, 1])) 



    plt.plot(x_data,mu_with_decay(x_data, *popt), label = country)

    plt.scatter(x_data, y_data)



    y_test = confirmed[confirmed['Country/Region'] == country].values[0][-len(x_data):]



    mu_params = popt

    timesteps = len(x_data)

    I0 = y_test[0]



    print(country,':')

    print('-----------')

    print('µ0:',round(popt[0], 3))

    print('lam:',round(popt[1], 3))

    print('b:',round(popt[2], 3))

    print('timesteps:', len(x_data))

    print('I0:', I0)

    print()

    

l = plt.legend()

l = plt.ylabel('Exponential Growth Rate (µ)')
#I0 = 15000

y0= [80e6, I0, 0]

t = np.arange(timesteps)



res = odeint(SIR, y0, t, args = (mu_params,))

extrap = odeint(SIR, y0, np.arange(100), args = (mu_params,))





plt.plot([a[1] for a in extrap], label = 'Infected extrapolated', linestyle=':')

plt.plot([a[1] for a in res], label = 'Infected')

#plt.plot([a[0] for a in res], label = 'Susceptible')

plt.plot([a[2] for a in extrap] , label = 'Removed')



y_test_deaths = deaths[deaths['Country/Region'] == country].values[0][-len(x_data):]

#plt.plot( y_test_deaths, label = 'Reported')



l = plt.legend()

#plt.plot(np.diff([y0[0] - a[0] for a in res]))
plt.figure(dpi = 144)

for country, color in zip(['Sweden', 'Norway', 'Finland', 'Denmark'],

                          ['blue', 'orange', 'red', 'green']):

    plt.plot( growth_rates.loc[country,:], linestyle='--', 

             color = color, label = f'µ_infected {country}')

    plt.plot(growth_rates_death.loc[country,:], 

             color = color, label = f'µ_deceased {country}')



l = plt.xticks(fontSize = 10, rotation = 60)

l = plt.ylabel('Exponetnial Growth Rate (µ), 7 day sliding window')

l = plt.title('Slowing of reported cases in Skandinavia')

l = plt.legend(bbox_to_anchor = (1,1))