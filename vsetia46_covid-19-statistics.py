#Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import datetime

#Data Sources

covid_data_confirmed = pd.read_csv('/kaggle/input/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

covid_data_recovered = pd.read_csv('/kaggle/input/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

covid_data_death = pd.read_csv('/kaggle/input/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

#helper functions

def isDate(input):

    isValid = True

    possibleDate = input.split('/')

    for val in possibleDate:

        if(not val.isnumeric()):

            isValid = False

            break

    

    return isValid



def unpivot_dataset(dataset):

    data_cols = dataset.columns

    date_cols = []

    identify_cols = []



    for col in data_cols:

        if(isDate(col)):

            date_cols.append(col)

        else:

            identify_cols.append(col)

        

    processed_data = pd.melt(dataset,value_vars=date_cols,id_vars=identify_cols,var_name='Date',value_name='Cases Count')

    processed_data['Date'] = processed_data['Date'].astype('datetime64')

    processed_data['Cases Count'] = processed_data['Cases Count'].astype('int64')

    return processed_data





def clean_data(dataset):

    columns = dataset.columns

    for column in columns:

        if(dataset[column].dtype == 'int64'):

            dataset[column] = dataset[column].fillna(0)

            

           

        elif(dataset[column].dtype == 'float64'):

            dataset[column] = dataset[column].fillna(0.0)

            if(column in ['Active','Confirmed','Recovered','Death']):

                    dataset[column] = dataset[column].astype(int)

            

        elif(dataset[column].dtype == 'object'):

            dataset[column] = dataset[column].fillna('Not Known')

            if(column in ['Country/Region','Province/State']):

                dataset[column] = dataset[column].astype(str)

        

        elif(dataset[column].dtype == 'datetime64[ns]'):

            dataset[column] = dataset[column].fillna(datetime.datetime(1970,1,1))

        

        else: 

            print(dataset[column].dtype + ' cleaning definition is not given')

            

            

        

    return dataset

    
#covid_data.columns

#data_description

#covid_data.shape

#covid_data.head()
#Combining Dataset of Confirmed, Recovered and Deaths

join_keys = ['Country/Region','Date','Province/State','Lat','Long']





processed_data_confirmed_cases = unpivot_dataset(covid_data_confirmed)

processed_data_confirmed_cases = processed_data_confirmed_cases.rename(columns={'Cases Count':'Confirmed'})

processed_data_confirmed_cases = processed_data_confirmed_cases.set_index(join_keys)



processed_data_recovered_cases = unpivot_dataset(covid_data_recovered)

processed_data_recovered_cases = processed_data_recovered_cases.rename(columns={'Cases Count':'Recovered'})

processed_data_recovered_cases = processed_data_recovered_cases.set_index(join_keys)



processed_data_death_cases = unpivot_dataset(covid_data_death)

processed_data_death_cases = processed_data_death_cases.rename(columns={'Cases Count':'Death'})

processed_data_death_cases = processed_data_death_cases.set_index(join_keys) 

#Combining Datasets

processed_cases = processed_data_confirmed_cases.join(processed_data_recovered_cases,lsuffix='_C',rsuffix='_R').join(processed_data_death_cases,lsuffix='_CR',rsuffix='_D')

#Active Cases Column

processed_cases['Active'] = processed_cases['Confirmed'] - processed_cases['Recovered'] - processed_cases['Death']



processed_cases = processed_cases.reset_index()
#Clean Data

processed_cases = clean_data(processed_cases)

latest_date = processed_cases['Date'].max()

latest_cases = processed_cases.loc[processed_cases['Date']==latest_date]



Active = latest_cases['Active'].sum()

Confirmed = latest_cases['Confirmed'].sum()

Recovered = latest_cases['Recovered'].sum()

Death = latest_cases['Death'].sum()

'Active: ' + str(Active)+' , Confirmed: '+str(Confirmed)+' , Recovered: '+str(Recovered)+' and Death: '+str(Death)


cases_overall = latest_cases.sort_values(by='Active',ascending=True) 

active_overall = list(cases_overall['Active'])

countries = list(cases_overall['Country/Region'])

plt = mpl.pyplot



fig, ax = plt.subplots(figsize=(20,300))

ax.barh(countries, active_overall)





ax.set(xlabel='Cases Count', ylabel='Country',

       title='COVID 19 Active Cases')

plt.style.use('fivethirtyeight')
check_trend_on_date = latest_date

country_name = 'India'
#Date That Can Be Changed To Get Date Wise Trends

viz_covid_data = processed_cases.loc[(processed_cases['Country/Region']==country_name) & (processed_cases['Date']==check_trend_on_date)]

viz_covid_data
quality_params = ""

for dataset in [covid_data_confirmed,covid_data_recovered,covid_data_death]:

    for column in dataset.columns:

        quality_params += 'Column: {} Missing: {}'.format(column,str(len(dataset[dataset[column].isnull() ])))

        quality_params += ' Inf: {}'.format(str(len(dataset.loc[dataset[column] == np.inf ])))

        quality_params += ' Zero: {}'.format(str(len(dataset.loc[dataset[column] == 0 ]))) 

        quality_params += '\n'

    

    quality_params+="\n\n"
print(quality_params) 