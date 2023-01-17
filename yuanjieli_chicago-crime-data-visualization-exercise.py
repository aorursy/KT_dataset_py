# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('seaborn')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
crime = pd.read_csv('../input/Crimes_-_2001_to_present.csv',error_bad_lines=False)
crime.shape
crime.head()
df = crime.copy()
df.head()
# convert dates to pandas datetime format

df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y %I:%M:%S %p')

# setting the index to be the date will help us a lot later on

df.index = pd.DatetimeIndex(df.Date)
plt.figure(figsize=(20,10))

df.resample('M').size().plot(legend=False)

plt.title('Number of crimes per month (2001 - 2017)')

plt.xlabel('Months')

plt.ylabel('Number of crimes')

plt.show()
plt.figure(figsize=(20,10))

df.resample('W').size().plot(legend=False)

plt.title('Number of crimes per week (2001 - 2017)')

plt.xlabel('Weeks')

plt.ylabel('Number of crimes')

plt.style.use(['seaborn-whitegrid','seaborn-paper'])

plt.show()
print(plt.style.available)
plt.figure(figsize=(20,10))

df.resample('D').size().rolling(365).sum().plot()

plt.title('Rolling sum of all crimes from 2001 - 2017')

plt.ylabel('Number of crimes')

plt.xlabel('Days')

plt.show()
crimes_count_date = df.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index=df.index.date, fill_value=0)

crimes_count_date.index = pd.DatetimeIndex(crimes_count_date.index)

plt.rcParams['ytick.labelsize'] = 11

plt.rcParams['xtick.labelsize'] = 11

plot = crimes_count_date.rolling(365).sum().plot(figsize=(12, 30), subplots=True, layout=(-1, 2), sharex=False, sharey=False)

plt.style.use(['seaborn-darkgrid'])
plt.figure(figsize=(20,10))

plt.ylabel('Months of the year', fontsize=20)

plt.xlabel('Number of crimes', fontsize=20)

plt.rcParams['ytick.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 14

plt.title('Number of crimes by month of the year')

df.groupby([df.index.month]).size().plot(kind='bar', style=["seaborn-whitegrid"])

plt.show()
plt.figure(figsize=(20,10))

plt.ylabel('Months of the year', fontsize=12)

plt.xlabel('Number of crimes', fontsize=12)

plt.rcParams['ytick.labelsize'] = 11

plt.rcParams['xtick.labelsize'] = 11

plt.title('Number of crimes by hour of the day')

df.groupby([df.index.hour]).size().plot(kind='bar', style=["seaborn-whitegrid"])

plt.show()
crimes_count_date = df.pivot_table('ID', aggfunc=np.size, columns='Arrest', index=df.index.date, fill_value=0)

crimes_count_date.index = pd.DatetimeIndex(crimes_count_date.index)

plot = crimes_count_date.rolling(365).sum().plot(figsize=(12, 5), subplots=True, layout=(-1, 2), sharex=False, sharey=False)

plt.style.use(['seaborn-darkgrid'])
crimes_count_date = df.pivot_table('ID', aggfunc=np.size, columns='Community Area', index=df.index.date, fill_value=0)

crimes_count_date.index = pd.DatetimeIndex(crimes_count_date.index)

plot = crimes_count_date.rolling(365).sum().plot(figsize=(12, 30), subplots=True, layout=(-1, 2), sharex=False, sharey=False)

plt.style.use(['seaborn-darkgrid'])
plt.figure(figsize=(20,10))

plt.rcParams['ytick.labelsize'] = 20

plt.rcParams['xtick.labelsize'] = 18

plt.ylabel('Frequency', fontsize=20)

plt.xlabel('Primary Type', fontsize=20)

crimeType = df.groupby([df['Primary Type']]).size().sort_values(ascending=False).plot(kind='bar', style='seaborn-paper')
df.groupby([df['Location Description']]).size().sort_values(ascending=False)
plt.figure(figsize=(20,100))

df.groupby([df['Location Description']]).size().sort_values(ascending=True).plot(kind='barh', style='seaborn-paper')

plt.title('Number of crimes by Urban type')

plt.ylabel('Crime by Urban Type')

plt.xlabel('Number of crimes')

plt.show()
plt.figure(figsize=(100,20))

plt.rcParams['ytick.labelsize'] = 40

plt.rcParams['xtick.labelsize'] = 36

plt.ylabel('Frequency', fontsize=40)

plt.xlabel('Primary Type', fontsize=40)

df.groupby([df['Community Area']]).size().sort_values(ascending=False).plot(kind='bar', style='seaborn-paper')

plt.title('Number of crimes by Community Area Number')

plt.ylabel('Crime by Community Area Number')

plt.xlabel('Number of crimes')

plt.show()
types = df.groupby([df['Primary Type']]).size().sort_values(ascending=False)

types
homicide = df[df["Primary Type"] == 'HOMICIDE'].copy()
homicide.head()
homicide.groupby([homicide['Location Description']]).size().sort_values(ascending=False)
plt.figure(figsize=(20,100))

homicide.groupby([homicide['Location Description']]).size().sort_values(ascending=True).plot(kind='barh', style='seaborn-paper')
plt.figure(figsize=(40,20))

plt.rcParams['ytick.labelsize'] = 24

plt.rcParams['xtick.labelsize'] = 24

plt.ylabel('Frequency', fontsize=30)

plt.xlabel('Primary Type', fontsize=30)

homicide.groupby([homicide['Community Area']]).size().sort_values(ascending=False).plot(kind='bar', style='seaborn-paper')

plt.title('Number of Homicide by Community Area')

plt.ylabel('Homicide by Community Area',size=30)

plt.xlabel('Homicide Distribution')



plt.show()
austin = homicide[homicide['Community Area'] == 25.0].copy()
plt.figure(figsize=(20,10))

plt.rcParams['ytick.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 14

plt.ylabel('Frequency', fontsize=20)

plt.xlabel('Primary Type', fontsize=20)

austin.resample('D').size().rolling(365).sum().plot()

plt.title('Rolling sum of homicide in Austin crimes from 2001 - 2017')

plt.ylabel('Number of crimes')

plt.xlabel('Years')

plt.show()
crimes_count_date = homicide.pivot_table('ID', aggfunc=np.size, columns='Community Area', index=homicide.index.date, fill_value=0)

crimes_count_date.index = pd.DatetimeIndex(crimes_count_date.index)

plot = crimes_count_date.rolling(365).sum().plot(figsize=(12, 30), subplots=True, layout=(-1, 2), sharex=False, sharey=False)

plt.style.use(['seaborn-darkgrid'])
hour_by_urbanType = df.pivot_table(values='ID', index='Location Description', columns=df.index.hour, aggfunc=np.size).fillna(0)

hour_by_location = df.pivot_table(values='ID', index='Community Area', columns=df.index.hour, aggfunc=np.size).fillna(0)

hour_by_type     = df.pivot_table(values='ID', index='Primary Type', columns=df.index.hour, aggfunc=np.size).fillna(0)

# hour_by_week     = df.pivot_table(values='ID', index=df.index.hour, columns=df.index.weekday_name, aggfunc=np.size).fillna(0)

# hour_by_week     = hour_by_week[days].T # just reorder columns according to the the order of days

location_by_type  = df.pivot_table(values='ID', index='Primary Type', columns='Community Area', aggfunc=np.size).fillna(0)

type_by_urbanType  = df.pivot_table(values='ID', index='Primary Type', columns='Location Description', aggfunc=np.size).fillna(0)
from sklearn.cluster import AgglomerativeClustering as AC



def scale_df(df,axis=0):

    '''

    A utility function to scale numerical values (z-scale) to have a mean of zero

    and a unit variance.

    '''

    return (df - df.mean(axis=axis)) / df.std(axis=axis)



def plot_hmap(df, ix=None, cmap='bwr'):

    '''

    A function to plot heatmaps that show temporal patterns

    '''

    if ix is None:

        ix = np.arange(df.shape[0])

    plt.imshow(df.iloc[ix,:], cmap=cmap)

    plt.colorbar(fraction=0.03)

    plt.yticks(np.arange(df.shape[0]), df.index[ix])

    plt.xticks(np.arange(df.shape[1]))

    plt.grid(False)

    plt.show()

    

def scale_and_plot(df, ix = None):

    '''

    A wrapper function to calculate the scaled values within each row of df and plot_hmap

    '''

    df_marginal_scaled = scale_df(df.T).T

    if ix is None:

        ix = AC(4).fit(df_marginal_scaled).labels_.argsort() # a trick to make better heatmaps

    cap = np.min([np.max(df_marginal_scaled.as_matrix()), np.abs(np.min(df_marginal_scaled.as_matrix()))])

    df_marginal_scaled = np.clip(df_marginal_scaled, -1*cap, cap)

    plot_hmap(df_marginal_scaled, ix=ix)

    

def normalize(df):

    result = df.copy()

    for feature_name in df.columns:

        max_value = df[feature_name].max()

        min_value = df[feature_name].min()

        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result
plt.figure(figsize=(15,60))

scale_and_plot(hour_by_urbanType)
plt.figure(figsize=(20,20))

scale_and_plot(hour_by_location)
plt.figure(figsize=(15,15))

scale_and_plot(hour_by_type)
plt.figure(figsize=(20,15))

scale_and_plot(location_by_type)
plt.figure(figsize=(100,25))

scale_and_plot(type_by_urbanType)