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
# importing the librabries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.model_selection import train_test_split

import missingno as msno # check missing value



# geographic visualization 

import chart_studio.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected= True)

df = pd.read_csv('../input/us-accidents/US_Accidents_May19.csv')
df.shape
df.columns
df.head()
# checking null values 

def chk_null(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = (df.isnull().sum()/df.shape[0]*100).sort_values(ascending = False)

    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    print(missing_data.head(10))
chk_null(df)
msno.matrix(df)
df.drop(['Precipitation(in)','Wind_Chill(F)','End_Lat','End_Lng'] ,axis =1,inplace = True)

df.shape
plt.figure(figsize=(10,7))

by_cat = df.groupby(["Source"]).size().sort_values(ascending = False)

sns.barplot(by_cat.values, by_cat.index.values, palette = "rocket")

plt.title("Data colleted by different surces")

plt.xlabel("Collection Count")
plt.figure(figsize=(10,7))

by_cat = df.groupby(["Timezone"]).size().sort_values(ascending = False)

sns.barplot(by_cat.values, by_cat.index.values, palette = "rocket")

plt.title("accident count for different timezone")

plt.xlabel("Numer of acidents")
plt.figure(figsize=(10,7))

by_cat = df.groupby(["Side"]).size().sort_values(ascending = False)

sns.barplot(by_cat.values, by_cat.index.values, palette = "rocket")

plt.title("accident count for different timezone")

plt.xlabel("Numer of acidents")
plt.figure(figsize=(7,10))

by_cat = df.groupby(['State']).size().sort_values(ascending = False)

sns.barplot(by_cat.values, by_cat.index.values, palette = "rocket")

plt.title("accident count for different State")

plt.xlabel("Numer of acidents")
# total number of accident grouped by US state 

acc_count = df.groupby('State')['State'].size()
data = dict(type = 'choropleth',

            locations = ["AL","AR","AZ","CA","CO","	CT","DC","DE","FL","GA","IA","ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI	","WV","WY"],

            locationmode = 'USA-states',

            colorscale= 'Electric',

            text= ["AL","AR","AZ","CA","CO","	CT","DC","DE","FL","GA","IA","ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI	","WV","WY"],

            z=acc_count,

            colorbar = {'title':'Accident_count'})

layout = dict(geo = {'scope':'usa'})

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
#chekcing corelation etween various features 

plt.figure(figsize=(8,8))

corr = df.corr()

sns.heatmap(corr)
# Number of unique classes in each 'object' column

# Number of each type of column

df.dtypes.value_counts()
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format="%Y/%m/%d %H:%M:%S")

df['DayOfWeekNum'] = df['Start_Time'].dt.dayofweek

df['DayOfWeek'] = df['Start_Time'].dt.weekday_name

df['MonthDayNum'] = df['Start_Time'].dt.day

df['HourOfDay'] = df['Start_Time'].dt.hour
sev_count = df.groupby('Severity').size()
df.Severity.value_counts(normalize=True).sort_index().plot.bar()

plt.grid()

plt.title('Severity')

plt.xlabel('Severity')

plt.ylabel('Fraction');
sns.set_style('whitegrid')

ax = sns.pointplot(x="HourOfDay", y="TMC", hue="DayOfWeek", data=df)

ax.set_title('hoursoffday vs TMC(Traffic Message Channel) of accident')

plt.show()
weekday = df.groupby('DayOfWeek').ID.count()

weekday = weekday/weekday.sum()

dayOfWeek=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

weekday[dayOfWeek].plot.bar()

plt.title('Acccidents by Weekday')

plt.xlabel('Weekday')

plt.ylabel('fraction of total accident');
st = pd.to_datetime(df.Start_Time, format='%Y-%m-%d %H:%M:%S')

end = pd.to_datetime(df.End_Time, format='%Y-%m-%d %H:%M:%S')
diff = (end-st)

top20 = diff.astype('timedelta64[m]').value_counts().nlargest(20)

print('top 20 accident durations correspond to {:.1f}% of the data'.format(top20.sum()*100/len(diff)))

(top20/top20.sum()).plot.bar(figsize=(7,5))

plt.title('Accident Duration [Minutes]')

plt.xlabel('Duration [minutes]')

plt.ylabel('Fraction');