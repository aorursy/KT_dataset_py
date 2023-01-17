# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install pandas --upgrade

import importlib

importlib.invalidate_caches()

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pd.__version__
url_cases = "https://dshs.texas.gov/coronavirus/TexasCOVID19DailyCountyCaseCountData.xlsx"

url_fatalities = "https://dshs.texas.gov/coronavirus/TexasCOVID19DailyCountyFatalityCountData.xlsx"

data = requests.get(url_cases, verify=False).content

df_cases = pd.read_excel(data, sheet_name="COVID-19 Cases",skiprows=[0,1]).dropna(how='any')

df_cases.columns = ["2020-{}".format(c.split("\n")[-1]) if c.find("Cases \n")>=0 else "_".join(c.lower().split()) for c in df_cases.columns]



data = requests.get(url_fatalities, verify=False).content

df_fatalities = pd.read_excel(data, sheet_name="COVID-19 Fatalities",skiprows=[0,1]).dropna(how='any')

df_fatalities.columns = ["2020-{}".format(c.split("\n")[-1]) if c.find("Fatalities")>=0 else "_".join(c.lower().split()) for c in df_fatalities.columns]
df_cases.sample(10)
df_fatalities.sample(10)
df_cases = df_cases.melt(id_vars=["county_name", "population"], 

        var_name="date", 

        value_name="total_cases")

df_fatalities = df_fatalities.melt(id_vars=["county_name", "population"], 

        var_name="date", 

        value_name="total_fatalities")



df_cases["date"] = pd.to_datetime(df_cases.date)

df_cases["total_cases"] = pd.to_numeric(df_cases.total_cases)

df_fatalities["date"] = pd.to_datetime(df_fatalities.date)

df_fatalities["total_fatalities"] = pd.to_numeric(df_fatalities.total_fatalities)

df_tx = pd.merge(df_cases, df_fatalities, how="left",on=["county_name","population","date"])

df_tx.fillna(0,inplace=True)



t = df_tx.set_index(["county_name","date"]).groupby(level=0)['total_cases'].diff()

df_tx["new_cases"] = t.values

t = df_tx.set_index(["county_name","date"]).groupby(level=0)['total_fatalities'].diff()

df_tx["new_fatalities"] = t.values
df_tx.sample(10)
# select counties to plot

counties = df_tx[df_tx.date==df_tx.date.max()].sort_values(by="total_cases",ascending=False)[:6].county_name.values
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

plt.xscale('log')

plt.yscale('log')

colors = list('rgbcmyk')

for ix, county in enumerate(counties):

  X = df_tx[(df_tx.county_name==county)&(df_tx.total_cases>0)].total_cases

  Y = df_tx[(df_tx.county_name==county)&(df_tx.total_cases>0)].new_cases

  p = Polynomial.fit(X, Y, 2)

  plt.plot(*p.linspace(),f'{colors[ix]}:',X,Y,f'{colors[ix]}.')

  

legend=[]

for c in counties:

  legend.append(f"{c if c != 'Total' else 'All'} county trend")

  legend.append(f"{c if c != 'Total' else 'All'} county data")

plt.legend(legend) 



plt.xlabel("Total Cases")

plt.ylabel("New Cases")

plt.title("New Cases vs Total Cases");

plt.minorticks_on()

plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
import ipywidgets as widgets



widgets.interact_manual.opts['manual_name'] = 'Update chart'

@widgets.interact_manual(

    county=widgets.Combobox(

    #value='Williamson',

    placeholder='Select a county',

    options=list(df_tx.county_name.unique()),

    description='County:',

    ensure_option=True,

    disabled=False

)

)

def plot(county):

  X = df_tx[df_tx.county_name==county]["date"]

  Y = df_tx[df_tx.county_name==county]["total_cases"]

  Y2 = df_tx[df_tx.county_name==county]["total_fatalities"]

  plt.figure(figsize=(15,5))



  plt.plot(X,Y,'.-',X,Y2,'.-')

  plt.legend(["total_cases","total_fatalities"])

  plt.text(X.max(),Y.max(),Y.max())

  plt.text(X.max(),Y2.max(),Y2.max())

  plt.xlabel("Date")

  plt.ylabel("#")

  plt.title(f"{county+' County' if county!='Total' else 'All counties'}");

  plt.minorticks_on()

  plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')

  plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')