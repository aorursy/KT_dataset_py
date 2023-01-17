# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import appropriate libraries
import pandas as pd
import numpy as np
import datetime as dt
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# Read the dataset
df_train = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df_train.head()
df_train.rename(columns={'ObservationDate':'Date', 'Province/State': 'Province_State', 'Country/Region':'Country_Region', 
                   'Last Update':'Last_Update','Deaths':'Fatalities'}, inplace=True)
df_train.columns
# Check the dataset
display(df_train.head())
display(df_train.describe())
display(df_train.info())
df_train.columns
df = df_train[['Date','Province_State','Country_Region','Last_Update','Confirmed','Fatalities','Recovered']]
df.head()
df.plot()
# number of Fatalities country wise
x=df.groupby(['Country_Region']).count()
x=x.sort_values(by='Fatalities',ascending=False)
x=x.iloc[0:10].reset_index()
x
# #plot
plt.figure(figsize=(10,6))
ax= sns.barplot(x.Country_Region, x.Confirmed, alpha=0.8)
plt.title("Fatalities per Country_Region")
plt.ylabel('# of Fatalities', fontsize=12)
plt.xlabel('Country_Region', fontsize=12)
plt.show()
# Plot to show Fatalities
df['Fatalities'].plot(legend=True,figsize=(10,4))
plt.show()
# Plot to show Country_wise Recovered
df['Recovered'].plot(legend=True,figsize=(10,4))
plt.show()
# Plot to show Confirmed
df['Confirmed'].plot(legend=True,figsize=(10,4))
plt.show()
# Plot to check Status of the different columns
df.plot(legend=True,figsize=(15,5))
plt.show()
# number of Confirmed cases per Country
x=df.groupby(['Country_Region', 'Province_State']).count()
x=x.sort_values(by='Confirmed',ascending=False)
x=x.iloc[0:200].reset_index()
x
# #plot
plt.figure(figsize=(16,8))
ax= sns.barplot(x.Country_Region, x.Fatalities, alpha=0.8)
plt.title("ConfirmedCases Country Wise")
plt.xlabel('# of Confirmed Cases', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.show()
# number of Confirmed cases per Country
x=df.groupby(['Country_Region', 'Fatalities']).count()
x=x.sort_values(by='Country_Region',ascending=False)
x=x.iloc[0:100].reset_index()
x
# #plot
plt.figure(figsize=(12,6))
ax= sns.barplot(x.Confirmed, x.Fatalities, alpha=0.8)
plt.title("ConfirmedCases Country Wise")
plt.xlabel('# of Confirmed Cases', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.show()
# number of Recovered cases per Country
x=df.groupby(['Country_Region', 'Recovered']).count()
x=x.sort_values(by='Confirmed',ascending=False)
x=x.iloc[0:10].reset_index()
x
# #plot
plt.figure(figsize=(12,6))
ax= sns.barplot(x.Recovered, x.Country_Region, alpha=0.8)
plt.title("ConfirmedCases Country Wise")
plt.xlabel('# of Recovered', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.show()
# How many countries affected 
countries = df['Country_Region'].unique()
print(f'{len(countries)} countries are in dataset:\n{countries}')
# number of Confirmed cases 
x=df.groupby(['Country_Region']).count()
x=x.sort_values(by='Confirmed',ascending=False)
x=x.iloc[0:10].reset_index()
x
# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.Confirmed, x.Fatalities, alpha=0.8)
plt.title("Infections Country wise")
plt.ylabel('Country', fontsize=12)
plt.xlabel('Confirmed', fontsize=12)
plt.show()
# number based on Province and Fatalities
x=df.groupby(['Province_State']).count()
x=x.sort_values(by='Fatalities',ascending=False)
x=x.iloc[0:100].reset_index()
x
# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.Confirmed, x.Recovered, alpha=0.8)
plt.title("Province State wise")
plt.ylabel('Recovered', fontsize=12)
plt.xlabel('Confirmed', fontsize=12)
plt.show()
# number based on County wise and Confirmed cases
x=df.groupby(['Country_Region']).count()
x=x.sort_values(by='Confirmed',ascending=False)
x=x.iloc[0:10].reset_index()
x
# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.Confirmed, x.Fatalities, alpha=0.8)
plt.title("County wise Fatalities")
plt.ylabel('Fatalities', fontsize=12)
plt.xlabel('Confirmed', fontsize=12)
plt.show()
# Plot to check Confirmed cases by Country 
df.Country_Region.value_counts().nlargest(20).plot(kind='bar', figsize=(10,5))
plt.title("Covid19 ConfirmedCases - Country_Region wise")
plt.ylabel("ConfirmedCases")
plt.xlabel("Ratio");
# Plot to check Confirmed cases 
df.Confirmed.value_counts().nlargest(20).plot(kind='bar', figsize=(10,5))
plt.title("Covid19 ConfirmedCases")
plt.ylabel("ConfirmedCases")
plt.xlabel("Ratio");
