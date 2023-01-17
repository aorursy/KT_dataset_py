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
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
df_test = pd.read_csv("../input/covid19-global-forecasting-week-5/test.csv")
df_train.head()
display(df_test.head())
display(df_train.describe())
display(df_train.info())
# If any null values 
df_train.isnull().sum()
# Check the countries impacted so far
countries = df_train['Country_Region'].unique()
print(f'{len(countries)} countries are in dataset:\n{countries}')
# number based on Province and Targets
x=df_train.groupby(['Province_State']).count()
x=x.sort_values(by='Target',ascending=False)
x=x.iloc[0:10].reset_index()
x
# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.Id, x.Id, alpha=0.8)
plt.title("Province State wise")
plt.ylabel('Target', fontsize=12)
plt.xlabel('Province_State', fontsize=12)
plt.show()
# number based on County wise and Targets
x=df_train.groupby(['County']).count()
x=x.sort_values(by='Target',ascending=False)
x=x.iloc[0:10].reset_index()
x
# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.Id, x.Id, alpha=0.8)
plt.title("County wise")
plt.ylabel('Target', fontsize=12)
plt.xlabel('County', fontsize=12)
plt.show()
# Plot to check Confirmed cases by County
df_train.County.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Covid19 ConfirmedCases - County wise")
plt.ylabel("ConfirmedCases")
plt.xlabel("Ratio");
# Plot to check Confirmed cases by Province_State
df_train.Province_State.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Covid19 ConfirmedCases - Province_State wise")
plt.ylabel("ConfirmedCases")
plt.xlabel("Ratio");
# Let's create a new dataframe with selected columns
df_new = df_train[['Date','Id','Country_Region','Population','Weight','Target','TargetValue']]
df_new.head()
# Plot to show TargetValue
df_new['TargetValue'].plot(legend=True,figsize=(10,4))
plt.show()
# Plot to show Country_wise Population
df_new['Population'].plot(legend=True,figsize=(10,4))
plt.show()
# Plot to show Weight
df_new['Weight'].plot(legend=True,figsize=(10,4))
plt.show()
# Plot to check Status of the different columns
df_new.plot(legend=True,figsize=(15,5))
plt.show()
# number of Confirmed cases per Country
x=df_new.groupby(['Country_Region', 'Target']).count()
x=x.sort_values(by='TargetValue',ascending=False)
x=x.iloc[0:10].reset_index()
x
# #plot
plt.figure(figsize=(12,6))
ax= sns.barplot(x.Country_Region, x.TargetValue, alpha=0.8)
plt.title("ConfirmedCases Country Wise")
plt.xlabel('# of Confirmed Cases', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.show()
#Let's check all the columns status
sns.pairplot(df_new)
# number of Targets based on Country
x=df_train.groupby(['Country_Region']).count()
x=x.sort_values(by='Target',ascending=False)
x=x.iloc[0:10].reset_index()
x
# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.Id, x.Id, alpha=0.8)
plt.title("Country_Region - Target")
plt.ylabel('Target', fontsize=12)
plt.xlabel('Countries', fontsize=12)
plt.show()
# Check by Confirmed and Fatalities
train = df_train.reset_index().groupby(['Country_Region', 'Target'])['TargetValue'].aggregate('first').unstack()
# Check the status of Confimedcases and Fatalities
sns.lmplot(x='ConfirmedCases', y = 'Fatalities', data = train)
plt.title('Country/Region wise')
# Plot to check ConfirmedCases and Fatalities
train.plot(legend=True,figsize=(15,5))
plt.show()
# Plot to show Confirmed cases
train['ConfirmedCases'].plot(legend=True,figsize=(15,5))
plt.show()
# Plot to check Confirmed cases by County for test dataset
df_test.County.value_counts().nlargest(20).plot(kind='bar', figsize=(10,5))
plt.title("Covid19 ConfirmedCases - County wise")
plt.ylabel("ConfirmedCases")
plt.xlabel("Ratio");
# Check the test dataset status
ax = df_test.loc[0:999].plot.area(stacked=False,alpha=0.3)
ax.legend(loc='center left',bbox_to_anchor=(1.0, 0.5));
