# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        dataframe = pd.read_csv(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.

China = dataframe[dataframe['Country/Region']=='China']

Italy = dataframe[dataframe['Country/Region']=='Italy']

US = dataframe[dataframe['Country/Region'] == 'US']

China.loc[:,'Date'] = pd.to_datetime(China.Date)

China = China.drop(['Lat','Long','Deaths'],axis=1)

ChinaConfirmed = China.groupby(China['Date'])

ChinaTotal = ChinaConfirmed['Confirmed'].sum()

ChinaTotal = ChinaTotal.to_frame()

ChinaTotal.columns = ['Total']

ChinaTotal['Changes'] = np.zeros(len(ChinaTotal),dtype=int)

ChinaTotal['Growth Factor'] = np.zeros(len(ChinaTotal),dtype=float)



Italy.loc[:,'Date'] = pd.to_datetime(Italy.Date)

Italy = Italy.drop(['Lat','Long','Deaths'],axis=1)

ItalyConfirmed = Italy.groupby(Italy['Date'])

ItalyTotal = ItalyConfirmed['Confirmed'].sum()

ItalyTotal = ItalyTotal.to_frame()

ItalyTotal.columns = ['Total']

ItalyTotal['Changes'] = np.zeros(len(ItalyTotal),dtype=int)

ItalyTotal['Growth Factor'] = np.zeros(len(ItalyTotal),dtype=float)



US.loc[:,'Date'] = pd.to_datetime(US.Date)

US = US.drop(['Lat','Long','Deaths'],axis=1)

USConfirmed = US.groupby(US['Date'])

USTotal = USConfirmed['Confirmed'].sum()

USTotal = USTotal.to_frame()

USTotal.columns = ['Total']

USTotal['Changes'] = np.zeros(len(USTotal),dtype=int)

USTotal['Growth Factor'] = np.zeros(len(USTotal),dtype=float)
ChinaTotal['Changes'] = ChinaTotal['Total'].diff()

ChinaTotal['Growth Factor'] = ChinaTotal.div(ChinaTotal['Changes'],axis=0)

for i in range(len(ChinaTotal)):

    ChinaTotal['Growth Factor'].iloc[i] = ChinaTotal['Changes'].iloc[i] / ChinaTotal['Changes'].iloc[i-1]

    

ItalyTotal['Changes'] = ItalyTotal['Total'].diff()

ItalyTotal['Growth Factor'] = ItalyTotal.div(ItalyTotal['Changes'],axis=0)

for i in range(len(ItalyTotal)):

    ItalyTotal['Growth Factor'].iloc[i] = ItalyTotal['Changes'].iloc[i] / ItalyTotal['Changes'].iloc[i-1]

    

USTotal['Changes'] = USTotal['Total'].diff()

USTotal['Growth Factor'] = USTotal.div(USTotal['Changes'],axis=0)

for i in range(len(USTotal)):

    USTotal['Growth Factor'].iloc[i] = USTotal['Changes'].iloc[i] / USTotal['Changes'].iloc[i-1]
%matplotlib inline

fig = plt.figure()



y0_values = ChinaTotal.loc[:, "Growth Factor"]

x0_values = np.linspace(1,len(ChinaTotal.loc[:, "Growth Factor"]),len(ChinaTotal.loc[:, "Growth Factor"]))

y1_values = ItalyTotal.loc[:, "Growth Factor"]

x1_values = np.linspace(1,len(ItalyTotal.loc[:, "Growth Factor"]),len(ItalyTotal.loc[:, "Growth Factor"]))

y2_values = USTotal.loc[:, "Growth Factor"]

x2_values = np.linspace(1,len(USTotal.loc[:, "Growth Factor"]),len(USTotal.loc[:, "Growth Factor"]))



poly_degree = 1



idx0 = np.isfinite(x0_values) & np.isfinite(y0_values)

coeffs0 = np.polyfit(x0_values[idx0], y0_values[idx0], 1)

idx1 = np.isfinite(x1_values) & np.isfinite(y1_values)

coeffs1 = np.polyfit(x1_values[idx1], y1_values[idx1], 1)

idx2 = np.isfinite(x2_values) & np.isfinite(y2_values)

coeffs2 = np.polyfit(x2_values[idx2], y2_values[idx2], 1)



poly_eqn0 = np.poly1d(coeffs0)

y0_hat = poly_eqn0(x0_values)

poly_eqn1 = np.poly1d(coeffs1)

y1_hat = poly_eqn1(x1_values)

poly_eqn2 = np.poly1d(coeffs2)

y2_hat = poly_eqn1(x2_values)



plt.figure(figsize = (30,10))

plt.ylim([0,5])

plt.plot_date(ChinaTotal.index, ChinaTotal["Growth Factor"], "ro")

plt.plot_date(ItalyTotal.index, ItalyTotal['Growth Factor'], "go")

plt.plot_date(USTotal.index, USTotal['Growth Factor'], "bo")



plt.plot(ChinaTotal.index,y0_hat,"-r")

plt.plot(ItalyTotal.index,y1_hat,"-g")

plt.plot(USTotal.index,y2_hat,"-b")



plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.show()

ChinaTotal.tail(5)
ItalyTotal.tail(5)
USTotal.tail(5)