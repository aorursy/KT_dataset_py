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
# Read input files

full_train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")

submission_data_orig = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/submission.csv")



train_data = full_train_data

train_end_date = max(train_data.Date)

train_data['Date'] = pd.to_datetime(train_data['Date'])

train_data.head()

train_end_date
# All imports

import matplotlib.pyplot as plt

from fbprophet import Prophet

from datetime import datetime

from datetime import timedelta

from sklearn.metrics import mean_squared_log_error



# Constants



# Total generated forecast

FORECAST_COUNT = 50



# Train data end date

TRAIN_DATA_END = train_end_date

# TODO Remove before final submission. PREDICTION_START_DATE is TRAIN_DATA_END + 1

# It exists like this because train data end and prediction start date are two different things at the moment

# Some places in the code filter data by train-data-end for understanding how much training data to read

# Some places set the starting value of a cell based on first day too

PREDICTION_START_DATE = '2020-03-26'



# TODO: Calculate this as opposed to eye estimate

AVRAGE_FATALITY_RATE_OF_INCREASE = 1.02

AVERAGE_CONFIRMED_CASES_RATE_OF_INCREASE = 1.05
# All Helper functions

def filter_train_data_for_public_leaderboard(data):

    """

    Filters data by dates and only keeps samples on or before 03-25-2020

    """

    data = data.loc[data.Date <= TRAIN_DATA_END]

    return data



def filter_data_by_date_range(data, start_date, end_date):

    """

    Filters data and keeps only those rows which lie in the date range left_date to right_date both inclusive

    """

    data = data.loc[((data.Date >= start_date) & (data.Date <= end_date))]

    return data



def remove_zero_cases(data, column_name):

    """

    Remove rows which have column_name in data DF set to 0

    """

    non_zero_columns_exp = data[column_name] != 0

    non_zero_columns = data[non_zero_columns_exp]

    return non_zero_columns



def filter_by_country(country, input_data):

    """

    Filters and returns rows in the dataframe belonging to the country pointed to by the country parameter

    """

    filtered_country_only = input_data["Country_Region"] == country

    return input_data[filtered_country_only]

    

def filter_by_state(state, input_data):

    """

    Call after filter_by_country since there is a slight possibility that two countries may have the same state

    """

    filtered_state_only = input_data["Province_State"] == state

    return input_data[filtered_state_only]



def prepare_data_for_prophet(input_df):

    """

    Prepare data for fitting into a Prophet forecasting model. That is timestamp (ds) and value (y) only.

    Input DataFrame is expected to contain the 'Date' and 'ConfirmedCases column.'

    """

    if 'ConfirmedCases' in input_df.columns:

        base_column = "ConfirmedCases"

    elif 'Fatalities' in input_df.columns:

        base_column = "Fatalities"

    else:

        raise ValueError('Either ConfirmedCases or Fatalities is expected in the DF')

        

    data = input_df[["Date", base_column]]

    # data = data.assign(InterpolateQuadratic=data.ConfirmedCases.interpolate(method='quadratic'))

    data_renamed = data.rename(columns = {'Date': 'ds', base_column: 'y'})

    return data_renamed



def group_by_state(country_data):

    """

    Returns a list of DF where one DF corresponds to one state.

    """

    states_data = list()

    for state in country_data.Province_State.unique():

        filtered_by_state = filter_by_state(state, country_data)

        if (len(filtered_by_state.index)) == 0:

            print('The {} state does not have any data'.format(state))

            continue

        states_data.append(filtered_by_state)

    return states_data



def rmsle(df1, df2):

    """

    Pass two DFs containing only one column. Name of the columns doesn't need to match.

    """

    return np.sqrt(mean_squared_log_error(df1, df2))



def filter_by_state(state, input_data):

    """

    Call after filter_by_country since there is a slight possibility that two countries may have the same state

    """

    filtered_state_only = input_data["Province_State"] == state

    return input_data[filtered_state_only]



def get_next_day(input_date):

    input_date = datetime.strptime(input_date, '%Y-%m-%d')

    next_date = input_date + timedelta(days=1)

    return datetime.strftime(next_date, format='%Y-%m-%d')



def get_previous_day(input_date):

    input_date = datetime.strptime(input_date, '%Y-%m-%d')

    next_date = input_date - timedelta(days=1)

    return datetime.strftime(next_date, format='%Y-%m-%d')
# Processing countries with states



def make_predictions_for_cc_and_f(state_data_relevant_cc_f, country, state):

    """

    Makes predictions for ConfirmedCases and Fatalities.

    Filters the training data state_data_relevant_cc_f to only include data until TRAIN_DATA_END.

    Creates predictions only from TRAIN_DATA_END + 1.

    """

    state_data_relevant_cc_f = state_data_relevant_cc_f[['Date', 'ConfirmedCases', 'Fatalities']]

    # Remove rows with 0 ConfirmedCases

    state_data_relevant_cc = remove_zero_cases(state_data_relevant_cc_f, 'ConfirmedCases')

    forecast_cc = pd.DataFrame()

    if len(state_data_relevant_cc) < 2:

    # This loop should never enter because by private testing, all countries will have at least one case

    # Writing it now since some countries still do not have data on 3-26

        state_data_relevant_cc = pd.DataFrame({

            'Date': [get_previous_day(get_previous_day(PREDICTION_START_DATE)), 

                     get_previous_day(PREDICTION_START_DATE)], 

            'ConfirmedCases': [0, 0],

            # Setting a slow rate of increase of death

            'Fatalities': [0.0, 0.0]

        })



    if len(state_data_relevant_cc) >= 2:

        # Add IncrementalConfirmedCases since prophet fits better on stationary data based on above observations

        state_data_relevant_stationary = state_data_relevant_cc[['ConfirmedCases']].diff()

        state_data_relevant_stationary.rename(columns={'ConfirmedCases':'IncrementalConfirmedCases'}, inplace=True)

        state_data_relevant_cc = state_data_relevant_cc.merge(

            state_data_relevant_stationary, left_index=True, right_index=True)

        state_data_relevant_cc[['IncrementalConfirmedCases']] = state_data_relevant_cc[[

            'IncrementalConfirmedCases']].fillna(value=0)

        # Renaming to ConfirmedCases since prepare_data_for_prophet function only understands ConfirmedCases or Fatalities

        prophet_input = prepare_data_for_prophet(state_data_relevant_cc[['Date', 'IncrementalConfirmedCases']]

                                     .rename(columns={'IncrementalConfirmedCases': 'ConfirmedCases'}))

        m_cc = Prophet()

        m_cc.fit(prophet_input)

        future = m_cc.make_future_dataframe(FORECAST_COUNT, freq='D')

        future = future.loc[future.ds >= PREDICTION_START_DATE]

        forecast_cc = m_cc.predict(future)

        forecast_cc = forecast_cc[['ds', 'yhat']]

        state_data_relevant_cc['Date'] = pd.to_datetime(state_data_relevant_cc['Date'], format='%Y-%m-%d')

        # Setting the first column value of yhat to be the count so far so that cumulative sum results in 

        # actual prediction

        last_train_row = state_data_relevant_cc[

            state_data_relevant_cc['Date'] == get_previous_day(PREDICTION_START_DATE)]

        if last_train_row.shape[0] != 0:

            last_cc = last_train_row.iloc[0]['ConfirmedCases']

        else:

            last_cc = 0

        forecast_cc.at[0, 'yhat'] = last_cc + forecast_cc.iloc[0]['yhat']

        # Removing all negative values, replacing with NaN and subsequently replacing NaNs with 0s

        forecast_cc = forecast_cc.assign(yhat = forecast_cc.yhat.where(forecast_cc.yhat.ge(0)))

        forecast_cc = forecast_cc.fillna(0)

        # Aggregating incremental counts to generate predicted confirmed cases

        forecast_cc['PredictedConfirmedCases'] = forecast_cc.yhat.cumsum()

        # Applying ceiling since fractional cases have no meaning

        forecast_cc['PredictedConfirmedCases'] = forecast_cc['PredictedConfirmedCases'].apply(np.rint)

        # forecast_cc has the columns: ds, yhat (incremental prediction), PredictedConfirmedCases

        # Renaming for better join output

        forecast_cc = forecast_cc[['ds', 'PredictedConfirmedCases']]

        forecast_cc.rename(columns={'ds': 'Date'}, inplace=True)



    # Fatalities logic here. Exactly same as above but for the column Fatalities

    state_data_relevant_f = remove_zero_cases(state_data_relevant_cc_f, 'Fatalities')

    # Need this check since removing 0s could have removed all rows as there are lot more 0 Fatalities

    forecast_f = pd.DataFrame()

    if len(state_data_relevant_f) < 2:

        # insert dummy row with linear increase. Since there are confirmed cases. This is going to do better than 0s

        # if one row exists, double that and insert. 

        # Otherwise insert one row with 1 and another with 2

        # We do not care about cc in this

        if len(state_data_relevant_f) == 1:

            f_value = state_data_relevant_f.iloc[0]['Fatalities']

        else:

            f_value = 0

        state_data_relevant_f = pd.DataFrame({

            'Date': [get_previous_day(get_previous_day(PREDICTION_START_DATE)), 

                     get_previous_day(PREDICTION_START_DATE)], 

            'ConfirmedCases': [0, 0],

            'Fatalities': [f_value, f_value * AVRAGE_FATALITY_RATE_OF_INCREASE]

        })

    

    state_data_relevant_stationary = state_data_relevant_f[['Fatalities']].diff()

    state_data_relevant_stationary.rename(columns={'Fatalities':'IncrementalFatalities'}, inplace=True)

    state_data_relevant_f = state_data_relevant_f.merge(state_data_relevant_stationary, left_index=True,

                                                        right_index=True)

    state_data_relevant_f[['IncrementalFatalities']] = state_data_relevant_f[[

        'IncrementalFatalities']].fillna(value=0)

    prophet_input = prepare_data_for_prophet(state_data_relevant_f[['Date', 'IncrementalFatalities']]

                                 .rename(columns={'IncrementalFatalities': 'Fatalities'}))

    m_f = Prophet()

    m_f.fit(prophet_input)

    future = m_f.make_future_dataframe(FORECAST_COUNT, freq='D')

    future = future.loc[future.ds >= PREDICTION_START_DATE]

    forecast_f = m_f.predict(future)

    forecast_f = forecast_f[['ds', 'yhat']]

    state_data_relevant_f['Date'] = pd.to_datetime(state_data_relevant_f['Date'], format='%Y-%m-%d')

    last_train_row = state_data_relevant_f[state_data_relevant_f['Date'] == get_previous_day(PREDICTION_START_DATE)]

    if last_train_row.shape[0] != 0:

        last_f = last_train_row.iloc[0]['Fatalities']

    else:

        last_f = 0

    forecast_f.at[0, 'yhat'] = last_f + forecast_f.iloc[0]['yhat']

    forecast_f = forecast_f.assign(yhat = forecast_f.yhat.where(forecast_f.yhat.ge(0)))

    forecast_f = forecast_f.fillna(0)

    forecast_f['PredictedFatalities'] = forecast_f.yhat.cumsum()

    forecast_f['PredictedFatalities'] = forecast_f['PredictedFatalities'].apply(np.rint)

    forecast_f = forecast_f[['ds', 'PredictedFatalities']]

    forecast_f.rename(columns={'ds': 'Date'}, inplace=True)



    forecast_final = pd.DataFrame()

    if forecast_f.empty and forecast_cc.count != 0:

        # impossible that forecast_cc is empty and forecast_f is not

        forecast_final = forecast_cc

    else:

        forecast_final = forecast_cc.merge(forecast_f, on='Date', how='left')

    forecast_final['Country_Region'] = country

    if state is not None:

        forecast_final['Province_State'] = state

    return forecast_final

# Iterate over every country

# For each country, prepare list of DF by state

# For each state DF, make FB prophet prediction into the future

# Create final DF of the form: Country, State, Date, Prediction

def make_predictions_for_countries_with_states(countries_with_state):

    """

    Makes predictions for one state at a time. Returns a DF containing the following columns:

    Country, State, Date, Prediction

    Parameters

    ----------

    countries_with_state : DataFrame

                           Containing all columns as pertaining in input_data but only those rows which have non 

                           NaN states

    """

    output_df = pd.DataFrame(columns=['Date', 'Country_Region', 

                                      'Province_State', 'PredictedConfirmedCases', 'PredictedFatalities'])

    for country in countries_with_state.Country_Region.unique():

        country_data = filter_by_country(country, countries_with_state)

        print('processing country {}'.format(country))

        states_data = group_by_state(country_data)

        for state_data in states_data:

            state = state_data.iloc[0].Province_State

            forecast_final = make_predictions_for_cc_and_f(state_data, country, state)

            output_df = pd.concat([output_df, forecast_final])

    print('finished processing all countries with state')        

    return output_df



def make_prediction_for_countries(countries_data):

    """

    Iterate over each country. Filter entries by country and perform prediction.

    """

    output_df = pd.DataFrame(columns=['Date', 'Country_Region', 

                                      'Province_State', 'PredictedConfirmedCases', 'PredictedFatalities'])

    for country in countries_data.Country_Region.unique():

        country_data = filter_by_country(country, countries_data)

        print('processing country {}'.format(country))

        forecast_final = make_predictions_for_cc_and_f(country_data, country, None)

        output_df = pd.concat([output_df, forecast_final])

    print('finished processing all countries which are without state')

    return output_df
# Run the predictions

# TODO Remove tests

train_data = filter_train_data_for_public_leaderboard(train_data)

countries_without_state = train_data[pd.isnull(train_data['Province_State'])]

# countries_without_state = filter_by_country('India', countries_without_state)

result_countries_without_state = make_prediction_for_countries(countries_without_state)

countries_with_state = train_data.dropna(subset=['Province_State'])

# countries_with_state = filter_by_country('Australia', countries_with_state)

result = make_predictions_for_countries_with_states(countries_with_state)
# Filling all missing values with 0s

result['PredictedFatalities'].fillna(0, inplace=True)

result['PredictedConfirmedCases'].fillna(0, inplace=True)

result_countries_without_state['PredictedFatalities'].fillna(0, inplace=True)

result_countries_without_state['PredictedConfirmedCases'].fillna(0, inplace=True)

final_result = pd.concat([result, result_countries_without_state])

final_result.head()

test_data.head()

# test_data = filter_test_data_for_public_leaderboard(test_data)

# max(test_data.Date) - 5/7

# max(final_result.Date) - 5/24

# state_data_relevant_cc['Date'] = pd.to_datetime(state_data_relevant_cc['Date'], format='%Y-%m-%d')

test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')



# Left join from test_data 

submission_data = test_data.merge(

    final_result, 

    left_on=['Date', 'Country_Region', 'Province_State'],

    right_on=['Date', 'Country_Region', 'Province_State'],

    how='left'

)



submission_data = submission_data[['ForecastId', 'PredictedConfirmedCases', 'PredictedFatalities']]

submission_data = submission_data.rename(columns={'PredictedConfirmedCases':'ConfirmedCases', 

                                                  'PredictedFatalities':'Fatalities'})

submission_data['Fatalities'].fillna(0, inplace=True)

submission_data['ConfirmedCases'].fillna(0, inplace=True)

submission_data.head(50)

# Publishing output for commit

submission_data.to_csv('submission.csv', index=False)
# validation_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

# validation_data['Date'] = pd.to_datetime(validation_data['Date'])

# validation_data = filter_data_by_date_range(validation_data, '2020-03-26', '2020-04-08')



# validation_data = filter_by_country('Australia', validation_data)

# validation_data = filter_by_state('New South Wales', validation_data)



# validation_data = validation_data.merge(

#     final_result, 

#     left_on=['Date', 'Country_Region', 'Province_State'],

#     right_on=['Date', 'Country_Region', 'Province_State'],

#     how='left'

# )

# # print(validation_data.head())

# validation_data.head(10)

# # Filter by countries and do rmsle one at a time iterating over uniqueCountries.
# validation_data.fillna(0, inplace=True)

# cc_score = rmsle(validation_data[['ConfirmedCases']], validation_data[['PredictedConfirmedCases']])

# f_score = rmsle(validation_data[['Fatalities']], validation_data[['PredictedFatalities']])





# print(cc_score)

# print(f_score)

# print((cc_score + f_score)/2)

# # 0.6580012040905447

# # 0.789636192992397

# # 0.7238186985414709



# # Australia with 0.4,0.45

# # 0.19140066882712942

# # 0.7522248023239893

# # 0.47181273557555936



# # Australia with 0, 0

# # 0.19140066882712942

# # 0.4371071402542409

# # 0.31425390454068514



# # Botswana with 0,0 

# # 0.6965142403895137

# # 0.543749384533382

# # 0.6201318124614479



# validation_data.head(50)
# import statistics 

# validation_data.head()



# # Iterate over all unique countries.

# # filter data by country

# # calculate rmsle for that country and keep adding two a DF containing the rows: Country, cc_score, f_score, combined

# output_df = pd.DataFrame()

# for country in validation_data.Country_Region.unique():

#     country_data = filter_by_country(country, validation_data)

#     cc_score = rmsle(country_data[['ConfirmedCases']], country_data[['PredictedConfirmedCases']])

#     f_score = rmsle(country_data[['Fatalities']], country_data[['PredictedFatalities']])

#     combined = statistics.median([cc_score, f_score])

#     row = pd.DataFrame({'Country_region': [country], 

#                        'cc_score': [cc_score], 

#                        'f_score': [f_score], 

#                        'combined': [combined]})

#     output_df = pd.concat([output_df, row], sort=False)

# output_df.head(50)

# by_cc = output_df.sort_values(by=['cc_score'], ascending=False)

# by_f = output_df.sort_values(by=['f_score'], ascending=False)

# by_combined = output_df.sort_values(by=['combined'], ascending=False)



# by_cc.to_csv('worst_cc_predicts_countries.csv', index=False)

# by_f.to_csv('worst_f_predicts_countries.csv', index=False)

# by_combined.to_csv('worst_combined_predict_countries.csv', index=False)



# by_combined.head(50)
# countries = final_result.Country_Region.unique()

# countries_data = train_data.loc[(

#     (train_data['Date'] == '2020-03-25') & (train_data['ConfirmedCases'] == 0)

# )]



# countries = final_result.Country_Region.unique()

# countries_data = full_train_data.loc[(

#     (train_data['Date'] == '2020-04-06') & (train_data['Fatalities'] == 0)

# )]





# countries_data.head(50)



# countries_data.shape

# 134



# countries_data for Fatalities = 0 on 3/25

# 11 rows only





# countries_data for ConfirmedCases = 0 on 3/25

# 11 rows only

# Resolution: Will be dealt with when full training data is available.



# All these countries have data now so for private leaderboard, predictions should be much better. 

# Nothing to worry here.

# Botswana

# Burma

# Burundi

# Northwest Territories	Canada

# Yukon	Canada

# NaN	Kosovo

# NaN	MS Zaandam

# NaN	Sierra Leone

# Anguilla	United Kingdom

# British Virgin Islands	United Kingdom

# Turks and Caicos Islands	United Kingdom
# # # Single country assessment

# pd.plotting.register_matplotlib_converters()



# # # Investigating how Niger can be improved



# # Generate training data

# country = 'Australia'

# train_data = filter_train_data_for_public_leaderboard(train_data)



# # For countries with state and country

# # niger_data = niger_data[pd.isnull(niger_data['Province_State'])]

# niger_data = filter_by_country(country, train_data)

# niger_data = filter_by_state('New South Wales', niger_data)



# # Make predictions

# niger_result = make_prediction_for_countries(niger_data)

# niger_result['Province_State'] = 'New South Wales'

# niger_result[['PredictedFatalities']] = niger_result[['PredictedFatalities']].fillna(0)

# niger_result[['PredictedConfirmedCases']] = niger_result[['PredictedConfirmedCases']].fillna(0)

# # print(max(niger_result.Date))



# # Generate validation data

# niger_validation_data = filter_by_country(country, 

#                                           pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv"))

# # For countries with state and country

# # niger_validation_data = niger_validation_data[pd.isnull(niger_validation_data['Province_State'])]

# niger_validation_data = filter_by_state('New South Wales', niger_validation_data)



# niger_validation_data['Date'] = pd.to_datetime(niger_validation_data['Date'])

# after_26 = niger_validation_data.loc[niger_validation_data.Date >= '2020-03-26']



# # after_26.head()

# # Merge forecast and validation data for comparison

# merged_result = after_26.merge(niger_result, how='left', on=['Date', 'Country_Region', 'Province_State'])

# # after_26.plot(x='Date', y='ConfirmedCases', figsize=(6,6), grid=True)

# # niger_result.plot(x='Date', y='PredictedConfirmedCases', figsize=(6,6), grid=True)

# merged_result.plot(x='Date', y=['ConfirmedCases', 'PredictedConfirmedCases'], legend=True, figsize=(6,6), grid=True)

# merged_result.plot(x='Date', y=['Fatalities', 'PredictedFatalities'], legend=True, figsize=(6,6), grid=True)

# plt.show()

# # print(rmsle(merged_result[['ConfirmedCases']], merged_result[['PredictedConfirmedCases']]))

# print(merged_result.head())

# merged_result.head()
