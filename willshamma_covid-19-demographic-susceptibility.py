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



from vega3 import VegaLite

import math

import IPython

import requests

import json

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
def config_browser_state():

  display(IPython.core.display.HTML('''

        <script src="/static/components/requirejs/require.js"></script>

        <script>

          window.outputs = [];

          requirejs.config({

            paths: {

              base: '/static/base',

              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',

            },

          });

        </script>

        <style>.vega-actions {display: none}</style>

        '''))



def vl(*args, **kw):

  config_browser_state()

  return VegaLite(*args, **kw)
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
agedata = pd.read_csv('../input/cdcagedatacovid/cdcAgeCovid19.csv')

agedata.columns = ['Age', 'Hospitalizations','ICUAdmission', 'Deaths']

agedata.dropna()
agedata = agedata.dropna()



#data cleaning to float values for analysis

agedata['Hospitalizations'] = ([float(j.split('–')[0])+float(j.split('–')[1])/2 for j in agedata.Hospitalizations])

agedata['ICUAdmission'] = ([0]+[float(j.split('–')[0])+float(j.split('–')[1])/2 for j in agedata.ICUAdmission[1:]])

agedata['Deaths'] = ([0]+ [float(j.split('–')[0])+float(j.split('–')[1])/2 for j in agedata.Deaths[1:]])



#converting values to percentages

agedata.Hospitalizations = [i/100 for i in agedata.Hospitalizations]

agedata.ICUAdmission = [i/100 for i in agedata.ICUAdmission]

agedata.Deaths = [i/100 for i in agedata.Deaths]

agedata
usaagedata = pd.read_csv('../input/census-age-data/agedata.csv', header= None)

usaagedata.columns = ['a','b','c','d','e','f','g','h']

usaagedatanew = usaagedata[['a','b','c']].dropna()[1:]

percentages =[usaagedatanew.c[:4].sum(), usaagedatanew.c[4:9].sum(), usaagedatanew.c[9:11].sum(), usaagedatanew.c[11:13].sum(), usaagedatanew.c[13:15].sum(), usaagedatanew.c[15:17].sum(), usaagedatanew.c[17:].sum()]

percentages = [i/100 for i in percentages]
#Rough estimate of the US Population with the 70% infection threshold (we assume only about 70% of people will contract the disease before the contagion subsides - a rough guide for infectious diseases)

USApop = 322000000*0.7

population = [i*USApop for i in percentages]

output  = pd.DataFrame({'Age': agedata.Age.values, 'Population Representation': population, 'Hospitalizations':(agedata.Hospitalizations.values * percentages * USApop), 'ICU':(agedata.ICUAdmission.values * percentages * USApop), 'Deaths':(agedata.Deaths.values * percentages * USApop)})

output.plot.bar(x='Age', y = ['Population Representation','Hospitalizations', 'ICU', 'Deaths'])
output.plot.bar(x='Age', y = ['Population Representation','Hospitalizations', 'ICU', 'Deaths'], log = True, legend = False)