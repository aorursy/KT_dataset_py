import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import os

import numpy as np

import datetime as dt

import scipy.optimize as opt

from sklearn.metrics import mean_squared_error

plt.rcParams['figure.max_open_warning'] = 0
### Parameters



# lockdown_window

#

# We will be plotting countries that have implemented a lockdown.

# This parameter selects the countries that will be plotted: those that

# have been locked down for a specified number of days



lockdown_window = 21 
# Load the data



# get_date_str

#

# Used to format dates for display



def get_date_str(d0,fmt='%d %b %Y'):

   return pd.to_datetime(d0).strftime(fmt)

    

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





df_train    = None

df_test     = None

df_lockdown = None



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_path = os.path.join(dirname,filename)

        if filename.startswith('train'):

            df_train = pd.read_csv(file_path, index_col='Id')

        if filename.startswith('test'):

            df_test = pd.read_csv(file_path)

        if filename.startswith('countryLockdowndates.csv'):

            df_lockdown = pd.read_csv(file_path)

 

print ('Data from {0} to {1}'.format(get_date_str(df_train.Date.min()),get_date_str(df_train.Date.max())))

# clean_up_gaps in df_train

#

# a function that deletes territories that contain gaps, 

# i.e. data was missing, so Johns Hopkins added a placeholder

#

def clean_up_gaps(row):

    print ('\t{0}-{1}'.format(row.Country_Region,row.Province_State))

    df_train.drop(df_train[(df_train.Country_Region==row.Country_Region) &

                           (df_train.Province_State==row.Province_State)].index, inplace=True)

    

df_train['Province_State'].fillna(' ',inplace=True)



by_ctry_prov = df_train.groupby(['Country_Region','Province_State'])[['ConfirmedCases','Fatalities']]

df_train[['NewCases','NewFatalities']]= by_ctry_prov.transform(lambda x: x.diff().fillna(0))



gapped_data = df_train[df_train.NewCases < 0]



if len(gapped_data.Country_Region)>0:

    print ("Deleting data where there are gaps")

    gapped_data.apply(clean_up_gaps,axis='columns')
# reporting_territory

#

# It will be useful to concatenate Country_Region and Province_State



def reporting_territory(row):

    if pd.isna(row.Province_State) or len(row.Province_State.strip())==0:

        return row.Country_Region

    else:

        return row.Country_Region + '-'+row.Province_State

    

df_train['Reporting_Territory'] = df_train.apply(reporting_territory,axis='columns')

df_train["Day"]                 = pd.to_datetime(df_train["Date"]).astype(int)
# Get rid of undefined fields

has_null        = df_lockdown.Date.isnull()|df_lockdown.Type.isnull()|df_lockdown.Reference.isnull()

rows_with_nulls = df_lockdown.loc[has_null,:].index

df_lockdown.drop(rows_with_nulls,inplace=True)

    

# Eliminate nans (especially from Province)

df_lockdown.replace(np.nan, '', regex=True,inplace=True)

print ('We have lockdown data for {0} countries'.format(df_lockdown.shape[0]))



# Harmonize column names with infection and death data



df_lockdown.rename(columns={"Country/Region":"Country_Region",

                            "Province": "Province_State"},

                   inplace=True)



df_lockdown['Reporting_Territory'] = df_lockdown.apply(reporting_territory,axis='columns')

    

    

# fix error in datafile (was '23/3030')

#df_lockdown.loc[df_lockdown.Reporting_Territory=='US-Ohio',['Date']]="23/03/2020"

 

df_lockdown["Day"]  = df_lockdown["Date"].map(lambda x:pd.to_datetime(x,format='%d/%m/%Y'), na_action='ignore')

# Delete lockdown records where there are gaps in infections & deaths

# so we don't get spurious mismatches



def remove_lockdown_gaps(row):

    df_lockdown.drop(

        df_lockdown[(df_lockdown.Country_Region==row.Country_Region) &

                         (df_lockdown.Province_State==row.Province_State)].index,

                    inplace=True)

    

if len(gapped_data.Country_Region)>0:

    print ("Deleting lockdown records where there are gaps in infections & deaths")

    gapped_data.apply(remove_lockdown_gaps,axis='columns')

    

df_territories = df_lockdown.merge(df_train,on=['Reporting_Territory','Reporting_Territory']).Reporting_Territory.unique()



print ('Unmatched Lockdown records')

for territory in df_lockdown[~df_lockdown.Reporting_Territory.isin(df_territories)].Reporting_Territory:

    print ('\t{0}'.format(territory))

    

print ('\nUnmatched Infection and Death records (treat these as not locked down)')

for territory in df_train[~df_train.Reporting_Territory.isin(df_territories)].Reporting_Territory.unique():

    print ('\t{0}'.format(territory))



# A few adhoc fixes for states

#     Guernsey

#     Jersey

#     Palestine

#     Vatican City

df_lockdown.loc[df_lockdown.Country_Region=='Guernsey',['Reporting_Territory']]="United Kingdom-Channel Islands"

df_lockdown.loc[df_lockdown.Country_Region=='Palestine',['Reporting_Territory']]="West Bank and Gaza"

df_lockdown.loc[df_lockdown.Country_Region=='Vatican City',['Reporting_Territory']]="Holy See"

# get_date_ticks

#

# This is used to declutter date access



def get_date_ticks(df,ngaps=7):

    dates    = df.Date.unique()

    n        = len(dates)-1

    stepsize = int(n/ngaps)

    return [dates[i] for i in range(0,n+1,stepsize)]



# get_date_range

#

# This is for the title



def get_date_range(df):

    return '{0} - {1}'.format(get_date_str(df.Date.min()),get_date_str(df.Date.max()))
# get_lockdown_status

#

# Find out whether specified territory has been locked down



def get_lockdown_status(territory):



    status = 'Not locked down'

    date   = None 



    rec = df_lockdown.loc[df_lockdown.Reporting_Territory==territory,"Day"]



    if len(rec)>0:

        status = 'Locked down'

        date = rec.values[0]

 

    return (status,date)
# get_title

#

# Generate title for a plot



def get_title(country_region,date_range,scaling='Confirmed'):

    status, date = get_lockdown_status(country_region)

    if status == 'Locked down':

        return '{0} Cases from {1}: {2}. {3} {4}.'.format(scaling,

                                                        country_region,

                                                        date_range,

                                                        status,

                                                        get_date_str(date))

    else:

        return '{0} Cases from {1}: {2}. {3}.'.format(scaling,

                                                        country_region,

                                                        date_range,

                                                        status)
# get_first_reported_case

#

# Find date that first case was deteced in a territory



def get_first_reported_case(territory):

   first_cases = df_train.loc[df_train.ConfirmedCases>0,["Reporting_Territory","Day"]].groupby(['Reporting_Territory']).min()

   return first_cases.loc[territory,['Day']].values[0]

# plot_country

#

# Plot data for specified territory



def plot_country(territory = 'New Zealand',save_figure=False):

    first_reported_case = get_first_reported_case(territory)

    country_data        = df_train.loc[(df_train.Reporting_Territory==territory) & (df_train.Day>=first_reported_case),:]



    _, lockdown_date    = get_lockdown_status(territory)



    dates       = country_data.Date

    cases       = country_data.ConfirmedCases

 

    locked_down = ["Locked down" if l else "Pre-lock down" for l in pd.to_datetime(country_data.Date) >= lockdown_date]

 

    plt.figure(figsize=(20,6))

    sns.set_palette("RdBu_r",1)

    

    # Workaround in case all data has been locked down or not

    

    if locked_down[0]!=locked_down[-1]:

        sns.scatterplot(x=dates,y=cases,hue=locked_down,style=locked_down,palette=['r','b'])

    else: 

        sns.scatterplot(x=dates,y=cases,hue=locked_down,style=locked_down,palette=['b'])

        

    plt.title(get_title(territory, get_date_range(country_data)))

    plt.xticks(get_date_ticks(country_data))

    

    if save_figure:

        plt.savefig(territory)



plot_country()

plot_country(territory='Italy')
def plot_country_ignore_error(territory,save_figure=False):

    try:

        plot_country(territory=territory.Reporting_Territory,save_figure=save_figure)

    except:

        print ('Failed to process {0}'.format(territory.Reporting_Territory))



cutoff_date = dt.datetime.today() -  dt.timedelta(days=lockdown_window)



_=df_lockdown.loc[df_lockdown.Day<cutoff_date,:].apply(plot_country_ignore_error,axis='columns')
def logistic_function(x, a, b, x0,d):

    return a/(1+np.exp(-b*(x-x0))) + d 







def fit_logistic(territory = 'China-Anhui'):

    first_reported_case = get_first_reported_case(territory)

    country_data        = df_train.loc[(df_train.Reporting_Territory==territory) & (df_train.Day>=first_reported_case),:]

    dates               = [d for d in range(len(country_data.Day.values))]

    final_cases         = country_data.ConfirmedCases.values.max()

    cases               = country_data.ConfirmedCases.values/final_cases

    (a, b, x0,d), _     = opt.curve_fit(logistic_function, dates, cases)

    prediction          = [final_cases* logistic_function(x, a, b, x0,d) for x in dates]

    rms_error           = np.sqrt(mean_squared_error(country_data.ConfirmedCases.values,prediction))

    

    plt.figure(figsize=(20,6))

    plt.plot(country_data.Date,country_data.ConfirmedCases,c='b',label='Confirmed Cases')

    plt.plot(country_data.Date,prediction,c='r',label='Predicted cases')

    plt.xticks(get_date_ticks(country_data))

    plt.suptitle('Fitting {0}'.format(territory))

    plt.title('a={0}, b={1}, x0={2}, d={3}. RMS error={4:.0f}'.format (a, b, round(x0),d, rms_error ))

    plt.legend()

    

fit_logistic()

#fit_logistic(territory = 'China-Fujian')

#fit_logistic(territory = 'China-Guangdong')

#fit_logistic(territory = 'New Zealand')

def fit_logistic_ignore_error(territory):

    try:

        fit_logistic(territory=territory.Reporting_Territory)

    except:

        print ('Failed to process {0}'.format(territory.Reporting_Territory))

        

_=df_lockdown.loc[df_lockdown.Day<cutoff_date,:].apply(fit_logistic_ignore_error,axis='columns')