import pandas as pd

import numpy as np

import matplotlib as plt

import seaborn as sns

%matplotlib inline
mydata = pd.read_csv('../input/drug_induced_deaths_1999-2015.csv')

mydata.head()
vis1 = sns.lmplot( data = mydata, x = 'Year', y = 'Deaths', fit_reg = False, hue = 'State', size = 10 )
from IPython.display import IFrame

IFrame('https://public.tableau.com/profile/jared.yu#!/vizhome/DrugDeathRateAnalytics0/Growth?publish=yes', width=900, height=550)
from IPython.display import IFrame

IFrame('https://public.tableau.com/profile/jared.yu#!/vizhome/DrugDeathRateAnalytics1/Correlation?publish=yes', width=900, height=550)
from IPython.display import IFrame

IFrame('https://public.tableau.com/profile/jared.yu#!/vizhome/DrugDeathRateAnalytics/Divergence?publish=yes', width=900, height=550)