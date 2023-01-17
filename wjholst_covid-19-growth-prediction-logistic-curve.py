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
#import IPython

#IPython.display.IFrame(<iframe width="650" height="400" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" title="2019-nCoV" src="/gisanddata.maps.arcgis.com/apps/Embed/index.html?webmap=14aa9e5660cf42b5b4b546dec6ceec7c&extent=77.3846,11.535,163.5174,52.8632&zoom=true&previewImage=false&scale=true&disable_scroll=true&theme=light"></iframe>)
from IPython.display import HTML



HTML('<div style="position:relative;height:0;padding-bottom:56.25%"><iframe src="https://www.youtube.com/embed/jmHbS8z57yI?ecver=2" width="640" height="360" frameborder="0" style="position:absolute;width:100%;height:100%;left:0" allowfullscreen></iframe></div>')
## install calmap

#! pip install calmap
# essential libraries

import json

import random

from urllib.request import urlopen



# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

#import calmap

import folium

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



# color pallette

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')



# html embedding

from IPython.display import Javascript

from IPython.core.display import display

from IPython.core.display import HTML
# list files

#!ls ../input/corona-virus-report

# https://www.kaggle.com/imdevskp/corona-virus-report
# importing datasets





full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

sub = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')

test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')
full_table[full_table['Province/State']=='California']

train
ca_by_state = train.copy()



train.columns

ca_by_state.columns =['Id', 'Province/State', 'Country/Region', 'Lat', 'Long', 'Date',

       'Confirmed', 'Deaths']

ca_by_state = ca_by_state[ca_by_state.Date >'2020-03-09']



ca_by_state['Active'] = ca_by_state.Confirmed - ca_by_state.Deaths

ca_by_state
print ('Last update of this dataset was ' + str(train.loc[len(train)-1]['Date']))

print ('Last update of the studay dataset was ' + str(full_table.loc[len(full_table)-1]['Date']))
#rates

dict = {

        'California':ca_by_state,

        #'United States': us_by_date,

}
def plots_by_country (country, country_name):



    temp = country



    # adding two more columns

    temp['No. of Deaths to 100 Confirmed Cases'] = round(temp['Deaths']/temp['Confirmed'], 3)*100

    # temp['No. of Recovered to 1 Death Case'] = round(temp['Recovered']/temp['Deaths'], 3)

    #print (temp)



    

    #print (temp.iloc[13]['Date'])

    last_date = temp.iloc[len(temp)-1]['Date']

    death_rate = temp[temp.Date ==last_date]['No. of Deaths to 100 Confirmed Cases']

    temp = temp.melt(id_vars='Date', value_vars=['No. of Deaths to 100 Confirmed Cases', ], 

                     var_name='Ratio', value_name='Value')



    #str(full_table.loc[len(full_table)-1]['Date'])



    fig = px.line(temp, x="Date", y="Value", color='Ratio', log_y=True, width=1000, height=700,

                  title=country_name + ' Recovery and Mortality Rate Over Time', color_discrete_sequence=[dth, rec])

    fig.show()

    return death_rate, 0

        

rates = []

for key, value in dict.items():

    death_rate, recovered_rate  = plots_by_country (value,key)

    rates.append ([key,np.float(death_rate),np.float(recovered_rate)]) 
import pylab

from scipy.optimize import curve_fit



def sigmoid(x, x0, k):

     y = 1 / (1 + np.exp(-k*(x-x0)))

     return y



def exp (x,a,b):

    y = a* np.exp(x*b)

    return y



def gaussian(x, a, x0, sigma):

    return a*np.exp(-(x-x0)**2/(2*sigma**2))



def growth_rate_over_time (f, country, attribute, title):

    ydata = country[attribute]

    



    xdata = list(range(len(ydata)))



    rates = []

    for i, x in enumerate(xdata):

        if i > 2:

#            print (xdata[:x+1])

#            print (ydata[:x+1])



            popt, pcov = curve_fit(f, xdata[:x+1], ydata[:x+1])

            rates.append (popt[1])

    rates = np.array(rates) 

    pylab.style.use('dark_background')

    pylab.figure(figsize=(12,8))

    xdata = np.array(xdata)

    #pylab.grid(True, linestyle='-', color='0.75')

    pylab.plot(xdata[3:]+1, 100*rates, 'o', linestyle='solid', label=attribute)

    #if fit_good:

    #    pylab.plot(x,y, label='fit')

    #pylab.ylim(0, ymax*1.05)

    #pylab.legend(loc='best')

    pylab.xlabel('Days Since Start')

    pylab.ylabel('Growth rate percentage ' + attribute)

    pylab.title(title + attribute, size = 15)

    pylab.show()

    

        

    



def plot_curve_fit (f, country, attribute, title, normalize = False, curve = 'Exp'):

    #print (country)

    #country = country[10:]

    fit_good = True

    ydata = country[attribute]

    #ydata = np.array(ydata)

    xdata = range(len(ydata))

    mu = np.mean(ydata)

    sigma = np.std(ydata)

    ymax = np.max(ydata)    

    if normalize:

        ydata_norm = ydata/ymax

    else:

        ydata_norm = ydata

    #f = sigmoid

    try:

        if curve == 'Gauss': # pass the mean and stddev

            popt, pcov = curve_fit(f, xdata, ydata_norm, p0 = [1, mu, sigma])

        else:    

            popt, pcov = curve_fit(f, xdata, ydata_norm,)

    except RuntimeError:

        print ('Exception - RuntimeError - could not fit curve')

        fit_good = False

    else:



        fit_good = True

        

    if fit_good:

        if curve == 'Exp':   

            print (key + ' -- Coefficients for y = a * e^(x*b)  are ' + str(popt))

            print ('Growth rate is now ' + str(round(popt[1],2)))

        elif curve == 'Gauss':

            print (key + ' -- Coefficients are ' + str(popt))

        else:   # sigmoid 

            print (key + ' -- Coefficients for y 1/(1 + e^(-k*(x-x0)))  are ' + str(popt))

            

        print ('Mean error for each coefficient: ' + str(np.sqrt(np.diag(pcov))/popt))

    else:

        print (key + ' -- Could not resolve coefficients ---')

    x = np.linspace(-1, len(ydata), 100)

    if fit_good:

        y = f(x, *popt)

        if normalize:

            y = y * ymax

    plt.style.use('dark_background')

    pylab.figure(figsize=(15,12)) 

    #pylab.grid(True, linestyle='-', color='0.75')

    pylab.plot(xdata, ydata, 'o', label=attribute)

    if fit_good:

        pylab.plot(x,y, label='fit')

    pylab.ylim(0, ymax*1.05)

    pylab.legend(loc='best')

    pylab.xlabel('Days Since Start')

    pylab.ylabel('Number of ' + attribute)

    pylab.title(title + attribute, size = 15)

    pylab.show()

for key, value in dict.items():

    if key in ["China",'Rest of China w/o Hubei']:

        pass

    else:

        #growth_rate_over_time (exp, value, 'Confirmed', "Growth Rate Percentage - ")

        growth_rate_over_time (exp, value, 'Confirmed', key + ' - Growth Rate Percentage for ',)

        #growth_rate_over_time (exp, value, 'Deaths', key + ' - Growth Curve for ',)

        #growth_rate_over_time (exp, value, 'Recovered', key + ' - Growth Curve for ',False)
round (72/35,2)
for key, value in dict.items():

    if key in ["China",'Rest of China w/o Hubei']:

        pass

    else:

        plot_curve_fit (exp, value, 'Confirmed', key + ' - Growth Curve for ',False,'Exp')

        plot_curve_fit (exp, value, 'Deaths', key + ' - Growth Curve for ',False,'Exp')

        #plot_curve_fit (exp, value, 'Recovered', key + ' - Growth Curve for ',False,'Exp')
#    plot_curve_fit (sigmoid, value, 'Recovered', key + ' - Logistic Growth Curve for ',True,'Logistic')
plot_curve_fit (gaussian, ca_by_state, 'Active', 'California' + ' - Curve for Cases ',False,'Gauss')
x0 = 33

k = 0.27

kd = 0.3

x0_death = 35

results = [] 

total_estimated = 500000

total_deaths = total_estimated * 0.15

for x in range(1,44):

    conf = int(total_estimated * sigmoid(x, x0, k))

    deaths = int(total_deaths * sigmoid(x, x0_death, kd))

    print ('Confirmed estimate for day ' + str(x) + ' - ' + str(conf))

    print ('Death estimate for day ' + str(x) + ' - ' + str(deaths))

    results.append([x,conf,deaths])
ca_by_state
r = pd.DataFrame(results)

r.columns = sub.columns

sub = r.copy()

sub

sub.to_csv("submission.csv", index=False)