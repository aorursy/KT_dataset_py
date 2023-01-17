# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/DOB_Job_Application_Filings.csv")
df.columns
df
df.dtypes
dfd = df.loc[df['Doc #'] == 1]
dfd
dfd['DOBRunDate'].dtype
dfd['DOBRunDate'] =  pd.to_datetime(dfd['DOBRunDate'])
dfa = dfd.sort_values('DOBRunDate').groupby('Job #').tail(1)

dfa
dfa['Pre- Filing Date'] =  pd.to_datetime(dfa['Pre- Filing Date'])
dfy = dfa.loc[dfa['Pre- Filing Date'].dt.year == 2018]

dfy[['Job #', 'Pre- Filing Date']]
dfy.shape
dfm = dfa.loc[df['Borough'] == 'MANHATTAN']

resv = ['R-1', 'R-2', 'R-3', 'RES']

dfresv = dfm.loc[df['Existing Occupancy'].isin(resv)]

dfresv[['Job #', 'Borough','Existing Occupancy']]
dfresv.shape
dfm.shape
np.divide(dfresv.shape[0], dfm.shape[0])
years = [2013, 2014, 2015, 2016, 2017, 2018]

dfyy = dfa.loc[dfa['Pre- Filing Date'].dt.year.isin(years)]

dfyy
dfchi = dfyy[['Borough', 'Fully Permitted']]

dfchi
dfchi['Borough'].value_counts()
dfchi['Fully Permitted'].fillna(0,inplace=True)

dfchi["Fully Permitted"]=dfchi["Fully Permitted"].apply(lambda x: 1 if x!=0 else 0)

dfchi
contingency_table = pd.crosstab(

    dfchi['Borough'],

    dfchi['Fully Permitted'],

    margins = True

)

contingency_table
f_obs = np.array([contingency_table.iloc[0][1:3].values,

                  contingency_table.iloc[3][1:3].values

                 ])

f_obs
from scipy import stats

stats.chi2_contingency(f_obs)[0:3]
dfo = dfa.loc[dfa['City Owned'] != 'Y']

dfo[['Job #', 'City Owned']]
dfo['Borough'].value_counts()
dfo['Borough'].value_counts(normalize=True)
np.divide(0.426309, 0.225453)
dfsq = pd.DataFrame(dfy.Borough.value_counts().reset_index().values, columns=["Borough", "ApplicationCounts"])

sqm = [22.82, 69.5, 108.1, 42.47, 58.69]

dfsq['area'] = sqm

dfsq['AppPerSqm'] = np.divide(dfsq['ApplicationCounts'], dfsq['area'])

dfsq
np.divide(1461.39, 307.496)
dfraa = dfa.loc[(dfa['Existing Occupancy'].isin(resv)) & (dfa['Job Type'] == 'A1')]

dfraa[['Job #', 'Existing Occupancy', 'Job Type', 'Existing Dwelling Units', 'Proposed Dwelling Units']]
dfraa.shape
dfa[['Existing Dwelling Units', 'Proposed Dwelling Units']].replace(r'^\s*$', np.nan, regex=True)

dfra = dfa.loc[(dfa['Existing Occupancy'].isin(resv)) & (dfa['Job Type'] == 'A1') & (dfa['Existing Dwelling Units'] != 'NaN') & (dfa['Proposed Dwelling Units'] != 'NaN')]
dfra['Existing Dwelling Units'].dtype
dfra['Proposed Dwelling Units'].dtype
dfra['Existing Dwelling Units'] = pd.to_numeric(dfra['Existing Dwelling Units'], errors='coerce')

dfra['Proposed Dwelling Units'] = pd.to_numeric(dfra['Proposed Dwelling Units'], errors='coerce')

dfra[['Job #', 'Existing Occupancy', 'Job Type', 'Existing Dwelling Units', 'Proposed Dwelling Units']]
incr = dfra['Existing Dwelling Units'] < dfra['Proposed Dwelling Units']

dfra['Increase'] = incr

dfra[['Job #', 'Existing Occupancy', 'Job Type', 'Existing Dwelling Units', 'Proposed Dwelling Units', 'Increase']]
drincr = dfra.loc[dfra['Increase'] == True]

drincr[['Job #', 'Existing Occupancy', 'Job Type', 'Existing Dwelling Units', 'Proposed Dwelling Units', 'Increase']]
drincr.shape
np.divide(8284,56426 )
from datetime import timedelta

dfyyb = dfyy.loc[dfyy['Borough'] == 'BROOKLYN']

dfyyb['Fully Permitted'] =  pd.to_datetime(dfyyb['Fully Permitted'])

dfyyb['Days'] = (dfyyb['Fully Permitted'] - dfyyb['Pre- Filing Date']) / timedelta (days=1)

dfyybf = dfyyb[['Borough', 'Pre- Filing Date', 'Fully Permitted', 'Days']].dropna()

dfyybf['year_of_PreF'] = dfyybf['Pre- Filing Date'].dt.year



dfyybf
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import statsmodels.api as sm



X = dfyybf['year_of_PreF']

Y = dfyybf['Days']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

predictions = model.predict(X)



model.summary()
sns.lmplot(x='year_of_PreF', y='Days', data = dfyybf[['year_of_PreF', 'Days']])
dfm['GIS_NTA_NAME'].value_counts()
bor = dfm.loc[dfm['GIS_NTA_NAME'] != 'park-cemetery-etc-Manhattan']

bor['GIS_NTA_NAME'].value_counts()
from skimage.measure import EllipseModel

from matplotlib.patches import Ellipse