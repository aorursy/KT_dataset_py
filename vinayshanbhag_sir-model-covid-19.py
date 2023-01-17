# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
owid_url = "https://covid.ourworldindata.org/data/ecdc/full_data.csv"
df_all = pd.read_csv(owid_url)
df_all["date"] = pd.to_datetime(df_all.date)
## Countries of interest
countries = ["United States","Italy","Spain","France","Germany","United Kingdom"]

df_all[(df_all.location!="World")&(df_all.date==df_all.date.max())].iloc[:,1:].sort_values(["total_cases"],ascending=False)[:10].style.background_gradient(cmap="Reds")
from numpy.polynomial import Polynomial 
plt.figure(figsize=(15,10))
plt.xscale('log')
plt.yscale('log')
for ix, country in enumerate(countries):
  X = df_all[df_all.location==country].total_cases
  Y = df_all[df_all.location==country].new_cases
  p = Polynomial.fit(X, Y, 2)
  plt.plot(*p.linspace(),'.')
  

plt.legend(countries) 

plt.xlabel("Total Cases")
plt.ylabel("New Cases")
plt.title("Growth Trajectories");
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
from numpy.polynomial import Polynomial 
plt.figure(figsize=(15,10))
plt.xscale('log')
plt.yscale('log')
for ix, country in enumerate(countries):
  X = df_all[df_all.location==country].total_deaths
  Y = df_all[df_all.location==country].new_deaths
  p = Polynomial.fit(X, Y, 2)
  plt.plot(*p.linspace(),'.')
  

plt.legend(countries) 

plt.xlabel("Total Fatalities")
plt.ylabel("New Fatalities")
plt.title("Growth Trajectories (Fatalities)");
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.figure(figsize=(15,5))
for country in countries:
    plt.plot(df_all[df_all.location==country].date,list(df_all[df_all.location==country].new_cases.values))
plt.legend(countries) 
plt.title("New Cases");
plt.xticks(rotation=90);
#plt.minorticks_on()
plt.grid(which='major', linestyle='--', linewidth='0.5', color='black')
#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.figure(figsize=(15,5))
for country in countries:
  plt.plot(df_all[df_all.location==country].date,df_all[df_all.location==country].new_deaths)
plt.legend(countries) 
plt.title("New Deaths");
plt.xticks(rotation=90);
#plt.minorticks_on()
plt.grid(which='major', linestyle='--', linewidth='0.5', color='black')
#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.figure(figsize=(15,5))
for country in countries:
  plt.plot(df_all[df_all.location==country].date.dt.date,df_all[df_all.location==country].total_cases)
plt.legend(countries) 
plt.title("Total Cases");
plt.xticks(rotation=90);
#plt.minorticks_on()
plt.grid(which='major', linestyle='--', linewidth='0.5', color='black')
#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.figure(figsize=(15,5))
for country in countries:
  plt.plot(df_all[df_all.location==country].date.dt.date,df_all[df_all.location==country].total_deaths)
plt.legend(countries) 
plt.title("Total Deaths");
plt.xticks(rotation=90);
#plt.minorticks_on()
plt.grid(which='major', linestyle='--', linewidth='0.5', color='black')
#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.figure(figsize=(15,5))
for country in countries:
    plt.plot(df_all[df_all.location==country].date.dt.date,df_all[df_all.location==country].total_deaths/df_all[df_all.location==country].total_cases)
plt.legend(countries) 
plt.title("Case Fatality Rate");
plt.xticks(rotation=90)
#plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.figure(figsize=(15,10))
c = 'rgbcmyk'
p = ['-','.-','--','.','*-','o-','_-']
l = []
for i in p:
  for j in list(c):
    l.append(f'{j}{i}')
for idx, ctry in enumerate(countries):
  temp = df_all[df_all.location==ctry].copy()
  temp = temp[temp.total_cases>0].copy()
  plt.plot(
    range(len(temp)),
    temp.total_cases,
    l[idx]
  )
plt.legend(countries);
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('# days since first case')
plt.ylabel('# known cases')
plt.title('Known cases by country since first case');
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.figure(figsize=(15,10))
c = 'rgbcmyk'
p = ['-','.-','--','.','*-','o-','_-']
l = []
for i in p:
  for j in list(c):
    l.append(f'{j}{i}')

for idx, ctry in enumerate(countries):
  temp = df_all[df_all.location==ctry].copy()
  temp = temp[temp.total_cases>0].copy()
  plt.plot(
    range(len(temp)),
    temp.new_deaths,  
    )
plt.legend(countries);

plt.xlabel('# days since first case')
plt.ylabel('# new deaths')
plt.title('New deaths by country since first case');
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
import ipywidgets as widgets
from scipy.integrate import odeint

def eqs(val,t, beta, gamma, nu=0.0):  
	S_t,I_t,R_t,A_t = val
	S = -beta*S_t*I_t + nu*R_t
	I = beta*S_t*I_t-gamma*I_t
	R = gamma*I_t - nu*R_t
	return [S,I,R,I+R]

widgets.interact_manual.opts['manual_name'] = 'Update chart'
@widgets.interact_manual(
    r_naught=widgets.FloatSlider(value=2.2,min=0.8,max=3.5,step=0.01,description='$R_0$'), 
    gamma=widgets.FloatSlider(value=0.3,min=0.1,max=0.9,step=0.01,description='$\\gamma$'),
    nu=widgets.FloatSlider(value=0.0,min=0.0,max=0.9,step=0.001,description='$\\nu$'),
    infected_0=widgets.IntSlider(value=-5,min=-7,max=-3,step=-1,description='$I_0={10^n}$; n:'),
    T=widgets.IntSlider(value=90,min=30,max=360,step=30,description='T')
)
def plot(r_naught=2.2, gamma=0.3, nu=0.03, infected_0=-5, T=90):
	beta = gamma*r_naught
	step = 1
	start = 0.0
	end = T
	infected_0 = 10**infected_0
	susceptible_0 = 1-infected_0
	recovered_0 = 0.0
	init_val = (susceptible_0, infected_0, recovered_0, infected_0+recovered_0)

	t_range = np.arange(start, end+step, step)
	RES = odeint(eqs,init_val,t_range,(beta, gamma, nu))

	fig = plt.figure(figsize=(15,10))
	plt.plot(RES[:,0],linestyle='--',linewidth=2)
	plt.plot(RES[:,1],linestyle='-',linewidth=3)
	plt.plot(RES[:,2],linestyle='--',linewidth=2)
	plt.plot(RES[:,3],linestyle=':',linewidth=2)
	plt.legend(['Susceptible','Infected','Removed','Infected+Removed'],loc='center left')
	plt.title(f"S-I-R Model\n($R_0$={r_naught:0.2f}, $\\beta$={beta:0.2f}, $\\gamma$={gamma:0.2f}, $\\nu$={nu:0.3f})");
	plt.minorticks_on()
	plt.grid(which='major', linestyle='--', linewidth='0.5', color='red')
	plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	plt.show()
    

plot()
gamma=0.22
r_naught=1.9
nu=0.0
T=120
country = "United States"
beta = gamma*r_naught
step = 1
start = 0.0
end = T
pop = 10_000_000
infected_0 = 1/pop
susceptible_0 = 1-infected_0
recovered_0 = 0.0
init_val = (susceptible_0, infected_0, recovered_0, infected_0+recovered_0)

t_range = np.arange(start, end+step, step)
RES = odeint(eqs,init_val,t_range,(beta, gamma, nu))
offset= 0
fig = plt.figure(figsize=(15,10))
plt.plot(range(offset,end+1),RES[offset:,0],linestyle='--',linewidth=2)
plt.plot(range(offset,end+1),RES[offset:,1],linestyle='-',linewidth=3)
plt.plot(range(offset,end+1),RES[offset:,2],linestyle='--',linewidth=2)
plt.plot(range(offset,end+1),RES[offset:,3],linestyle=':',linewidth=2)
est_cases = (df_all[(df_all.location==country)&(df_all.total_cases>0)].total_cases*10/pop).values[offset:]
plt.plot(
    range(offset, offset+len(est_cases)),
    est_cases,
    'mo'
)
plt.legend(['Susceptible','Infected','Removed','Infected+Removed','Est. infections (10x confirmed cases)'],loc='center left')
plt.title(f"S-I-R Model ({country})\n($R_0$={r_naught:0.2f}, $\\beta$={beta:0.2f}, $\\gamma$={gamma:0.2f}, $\\nu$={nu:0.3f})");
plt.minorticks_on()
plt.grid(which='major', linestyle='--', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
