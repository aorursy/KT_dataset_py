import os

import pandas as pd

import numpy as np

import requests

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go



from datetime import date, timedelta, datetime

from fbprophet import Prophet

from fbprophet.make_holidays import make_holidays_df

from fbprophet.diagnostics import cross_validation, performance_metrics

from fbprophet.plot import plot_cross_validation_metric

import holidays



import pycountry

import plotly.express as px

from collections import namedtuple



import warnings

warnings.simplefilter('ignore')
# Thanks https://github.com/CSSEGISandData/COVID-19

# Thanks https://www.kaggle.com/corochann/covid-19-current-situation-on-august

for filename in ['time_series_covid19_confirmed_global.csv']:

    print(f'Downloading {filename}')

    url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/{filename}'

    myfile = requests.get(url)

    open(filename, 'wb').write(myfile.content)

    

confirmed_global_df = pd.read_csv('time_series_covid19_confirmed_global.csv')
# Thanks to https://www.kaggle.com/corochann/covid-19-current-situation-on-august

def _convert_date_str(df):

    try:

        df.columns = list(df.columns[:4]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in df.columns[4:]]

    except:

        print('_convert_date_str failed with %y, try %Y')

        df.columns = list(df.columns[:4]) + [datetime.strptime(d, "%m/%d/%Y").date().strftime("%Y-%m-%d") for d in df.columns[4:]]



_convert_date_str(confirmed_global_df)

confirmed_global_df
# Thanks to https://www.kaggle.com/corochann/covid-19-current-situation-on-august

df = confirmed_global_df.melt(

    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='ConfirmedCases')
df
df["Country/Region"].unique()
# Convert name of countries to ISO 3166

df["Country/Region"].replace({'Korea, South': 'Korea, Republic of'}, inplace=True)

df["Country/Region"].replace({'Russia': 'Russian Federation'}, inplace=True)

df["Country/Region"].replace({'US': 'United States'}, inplace=True)
df
df2 = df.groupby(["Date", "Country/Region"])[['Date', 'Country/Region', 'ConfirmedCases']].sum().reset_index()
df_countries = df2['Country/Region'].unique()

df_countries
latest_date = df2['Date'].max()

latest_date
# Thanks to dataset https://www.kaggle.com/vbmokin/covid19-holidays-of-countries

holidays_df = pd.read_csv('../input/covid19-holidays-of-countries/holidays_df_of_67_countries_for_covid_19.csv')

holidays_df
holidays_df['country'].unique()
holidays_df_code_countries = holidays_df['code'].unique()

holidays_df_code_countries
def dict_code_countries_with_holidays(list_name_countries: list,

                                      holidays_df: pd.DataFrame()):

    """

    Defines a dictionary with the names of user countries and their two-letter codes (ISO 3166) 

    in the dataset "COVID-19: Holidays of countries" 

    

    Returns: 

    - countries: dictionary with the names of user countries and their two-letter codes (ISO 3166) 

    - holidays_df_identificated: DataFrame with holidays data for countries from dictionary 'countries'

    

    Args: 

    - list_name_countries: list of the name of countries (name or common_name or official_name or alha2 or alpha3 codes from ISO 3166)

    - holidays_df: DataFrame with holidays "COVID-19: Holidays of countries"

    """

    

    import pycountry

    

    # Identification of countries for which there are names according to ISO

    countries = {}

    dataset_all_countries = list(holidays_df['code'].unique())

    list_name_countries_identificated = []

    list_name_countries_not_identificated = []

    for country in list_name_countries:

        try: 

            country_id = pycountry.countries.get(alpha_2=country)

            if country_id.alpha_2 in dataset_all_countries:

                countries[country] = country_id.alpha_2

        except AttributeError:

            try: 

                country_id = pycountry.countries.get(name=country)

                if country_id.alpha_2 in dataset_all_countries:

                    countries[country] = country_id.alpha_2

            except AttributeError:

                try: 

                    country_id = pycountry.countries.get(official_name=country)

                    if country_id.alpha_2 in dataset_all_countries:

                        countries[country] = country_id.alpha_2

                except AttributeError:

                    try: 

                        country_id = pycountry.countries.get(common_name=country)

                        if country_id.alpha_2 in dataset_all_countries:

                            countries[country] = country_id.alpha_2

                    except AttributeError:

                        try: 

                            country_id = pycountry.countries.get(alpha_3=country)

                            if country_id.alpha_2 in dataset_all_countries:

                                countries[country] = country_id.alpha_2

                        except AttributeError:

                            list_name_countries_not_identificated.append(country)

    holidays_df_identificated = holidays_df[holidays_df['code'].isin(countries.values())]

    

    print(f'Thus, the dataset has holidays in {len(countries)} countries from your list with {len(list_name_countries)} countries')

    if len(countries) == len(dataset_all_countries):

        print('All available in this dataset holiday data is used')

    else:

        print("Holidays are available in the dataset for such countries (if there are countries from your list, then it's recommended making changes to the list)")

        print(np.array(holidays_df[~holidays_df['code'].isin(countries.values())].country_official_name.unique()))

        

    return countries, holidays_df_identificated
countries_dict, holidays_df = dict_code_countries_with_holidays(df_countries,holidays_df)
def adaption_df_to_holidays_df_for_prophet(df, col, countries_dict):

    # Adaptation the dataframe df (by column=col) to holidays_df by list of countries in dictionary countries_dict

    

    # Filter df for countries which there are in the dataset with holidays

    df = df[df[col].isin(list(countries_dict.keys()))].reset_index(drop=True)

    

    # Add alpha_2 (code from ISO 3166) for each country

    df['iso_alpha'] = None

    for key, value in countries_dict.items():

        df.loc[df[col] == key, 'iso_alpha'] = value    

    

    return df
holidays_df
df2 = adaption_df_to_holidays_df_for_prophet(df2, 'Country/Region', countries_dict)

df2.columns = ['Date', 'Country', 'Confirmed', 'iso_alpha']

df2
print("Number of countries/regions with data: " + str(len(df2.Country.unique())))
df2.describe()
df2.head()
df2.tail()
lower_window_list = [0, -1, -2, -3] # must be exactly 4 values (identical allowed)

upper_window_list = [0, 1, 2, 3] # must be exactly 4 values (identical allowed)

prior_scale_list = [0.05, 0.5, 1, 15] # must be exactly 4 values (identical allowed)
def convert10_base4(n):

    # convert decimal to base 4

    alphabet = "0123"

    if n < 4:

        return alphabet[n]

    else:

        return (convert10_base4(n // 4) + alphabet[n % 4]).format('4f')
days_to_forecast = 3 # in future (after training data)

days_to_forecast_for_evalution = 14 # on the latest training data - for model training

first_forecasted_date = sorted(list(set(df2['Date'].values)))[-days_to_forecast]



print('The first date to perform forecasts for evaluation is: ' + str(first_forecasted_date))
print('The end date to perform forecasts in future for is: ' + (datetime.strptime(df2['Date'].max(), "%Y-%m-%d")+timedelta(days = days_to_forecast)).strftime("%Y-%m-%d"))
confirmed_df = df2[['Date', 'Country', 'Confirmed', 'iso_alpha']]

confirmed_df
all_countries = confirmed_df['Country'].unique()

all_countries
n = 64 # number of combination of parameters lower_window / upper_window / prior_scale
def make_forecasts(all_countries, confirmed_df, holidays_df, days_to_forecast, days_to_forecast_for_evalution, first_forecasted_date):

    # Thanks to https://www.kaggle.com/vbmokin/covid-19-in-ukraine-prophet-holidays-tuning

    

    def eval_error(forecast_df, country_df_val, first_forecasted_date, title):

        # Evaluate forecasts with validation set val_df and calculaction and printing with title the relative error

        forecast_df[forecast_df['yhat'] < 0]['yhat'] = 0

        result_df = forecast_df[(forecast_df['ds'] >= pd.to_datetime(first_forecasted_date))]

        result_val_df = result_df.merge(country_df_val, on=['ds'])

        result_val_df['rel_diff'] = (result_val_df['y'] - result_val_df['yhat']).round().abs()

        relative_error = [sum(result_val_df['rel_diff'].values)*100/result_val_df['y'].sum()]

        return relative_error

    

    def model_training_forecasting(df, forecast_days, holidays_df=None):

        # Prophet model training and forecasting

        

        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False, 

                        holidays=holidays_df, changepoint_range=1, changepoint_prior_scale = 0.25)

        model.add_seasonality(name='weekly', period=7, fourier_order=8, mode = 'multiplicative', prior_scale = 0.3)

        #model.add_seasonality(name='triply', period=3, fourier_order=2, mode = 'multiplicative', prior_scale = 0.5)

        model.fit(df)

        future = model.make_future_dataframe(periods=forecast_days)

        forecast = model.predict(future)

        forecast[forecast['yhat'] < 0]['yhat'] = 0

        return model, forecast

    

    forecast_dfs = []

    relative_errors = []

    cols_w = ['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper', 'additive_terms', 'additive_terms_lower', 'additive_terms_upper',

              'multiplicative_terms','multiplicative_terms_lower', 'multiplicative_terms_upper', 'weekly', 'weekly_lower', 'weekly_upper']

    cols_h = ['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper', 'additive_terms', 'additive_terms_lower', 'additive_terms_upper',

              'holidays', 'holidays_lower', 'holidays_upper', 'multiplicative_terms','multiplicative_terms_lower', 'multiplicative_terms_upper', 'weekly',

              'weekly_lower', 'weekly_upper']

    relative_errors_holidays = []

    counter = 0

    results = pd.DataFrame(columns=['Country', 'Country_code', 'Conf_real', 'Conf_pred', 'Conf_pred_h', 'n_h', 'err', 'err_h', 'lower_window', 'upper_window', 'prior_scale', 'how_less, %'])

    

    

    for j in range(len(all_countries)):

        country = all_countries[j]

        if country in confirmed_df['Country'].values:

            print(f'Country {str(country)} is listed')

            country_df = confirmed_df[confirmed_df['Country'] == country].reset_index(drop=True)

            country_iso_alpha = country_df['iso_alpha'][0]

            

            # Calc daily values

            #country_df['Confirmed'] = country_df['Confirmed'].diff()

            #country_df.loc[0,'Confirmed'] = 0

            

            # Selection holidays of country

            country_holidays_df = holidays_df[holidays_df['code'] == country_iso_alpha][['ds', 'holiday', 'lower_window', 'upper_window', 'prior_scale']].reset_index(drop=True)

            country_dfs = []            

            

            # Data preparation for forecast with Prophet

            country_df = country_df[['Date', 'Confirmed']]

            country_df.columns = ['ds','y']

            country_df['ds'] = pd.to_datetime(country_df['ds'])



            # Set training and validation datasets

            country_df_future = country_df.copy()

            country_df_val = country_df[(country_df['ds'] >= pd.to_datetime(first_forecasted_date))].copy()

            country_df = country_df[(country_df['ds'] < pd.to_datetime(first_forecasted_date))]



            # Without holidays

            # Model training and forecasting without holidays

            model, forecast = model_training_forecasting(country_df, days_to_forecast_for_evalution)

            #fig1 = model.plot_components(forecast)



            # Evaluate forecasts with validation set val_df and calculaction and printing the relative error

            forecast_df = forecast[['ds', 'yhat']].copy()

            relative_errors += eval_error(forecast_df, country_df_val, first_forecasted_date, 'without holidays')



            # With holidays

            # Model training with tuning prior_scale and forecasting

            relative_error_holidays_min = relative_errors[-1]

            number_holidays = len(country_holidays_df[(country_holidays_df['ds'] > '2020-01-21') & (country_holidays_df['ds'] < '2020-10-01')])

            for i in range(n):

                if country_iso_alpha == 'il':

                    lower_window_i = 0

                    upper_window_i = 0

                    prior_scale_i = 10

                    i = 63

                else:

                    parameters_iter = convert10_base4(i).zfill(3)

                    lower_window_i = lower_window_list[int(parameters_iter[0])]

                    upper_window_i = upper_window_list[int(parameters_iter[1])]

                    prior_scale_i = prior_scale_list[int(parameters_iter[2])]

                country_holidays_df['lower_window'] = lower_window_i

                country_holidays_df['upper_window'] = upper_window_i

                country_holidays_df['prior_scale'] = prior_scale_i

                model_holidays, forecast_holidays = model_training_forecasting(country_df, days_to_forecast_for_evalution, country_holidays_df)

                

                # Evaluate forecasts with validation set val_df and calculaction and printing the relative error

                forecast_holidays_df = forecast_holidays[['ds', 'yhat']].copy()

                relative_error_holidays = eval_error(forecast_holidays_df, country_df_val, first_forecasted_date, 'with holidays impact')

                

                # Save results

                if country_iso_alpha == 'il':

                    relative_error_holidays_min = relative_error_holidays

                    forecast_holidays_df_best = forecast_holidays[cols_h]

                    model_holidays_best = model_holidays

                    lower_window_best = lower_window_i

                    upper_window_best = upper_window_i

                    prior_scale_best = prior_scale_i

                elif i == 0:

                    relative_error_holidays_min = relative_error_holidays

                    forecast_holidays_df_best = forecast_holidays[cols_h]

                    model_holidays_best = model_holidays

                    lower_window_best = lower_window_i

                    upper_window_best = upper_window_i

                    prior_scale_best = prior_scale_i

                elif (relative_error_holidays[0] < relative_error_holidays_min[0]):

                    relative_error_holidays_min = relative_error_holidays

                    forecast_holidays_df_best = forecast_holidays[cols_h]

                    model_holidays_best = model_holidays

                    lower_window_best = lower_window_i

                    upper_window_best = upper_window_i

                    prior_scale_best = prior_scale_i

                print('i =',i,' from',n,':  lower_window =', lower_window_i, 'upper_window =',upper_window_i, 'prior_scale =', prior_scale_i)

                print('error_holidays =',relative_error_holidays[0], 'err_holidays_min (WAPE)',relative_error_holidays_min[0], '\n')

            

            # Results visualization

            print('The best errors of model with holidays is', relative_error_holidays_min[0], 'with lower_window =', str(lower_window_best),

              ' upper_window =', str(upper_window_best), ' prior_scale =', str(prior_scale_best))

            print('The best errors WAPE of model with holidays is', relative_error_holidays_min[0], '\n')

            relative_errors_holidays += relative_error_holidays_min            



            # Save results to dataframe with all dates

            forecast_holidays_df_best['country'] = country

            forecast_holidays_df_best.rename(columns={'yhat':'confirmed'}, inplace=True)

            if j == 0:                

                forecast_holidays_dfs = forecast_holidays_df_best.tail(days_to_forecast_for_evalution)

            else:

                forecast_holidays_dfs = pd.concat([forecast_future_dfs, forecast_holidays_df_best.tail(days_to_forecast_for_evalution)])



            # Forecasting the future

            if relative_errors[-1] < relative_errors_holidays[-1]:

                # The forecast without taking into account the holidays is the best

                model_future_best, forecast_future_best = model_training_forecasting(country_df_future, days_to_forecast)

                forecast_plot = model_future_best.plot(forecast_future_best, ylabel='Confirmed in '+ country + ' (forecasting without holidays)')

                cols = cols_w

            else:

                # The forecast taking into account the holidays is the best

                country_holidays_df['prior_scale'] = prior_scale_best

                model_future_best, forecast_future_best = model_training_forecasting(country_df_future, days_to_forecast, country_holidays_df)

                forecast_plot = model_future_best.plot(forecast_future_best, ylabel='Confirmed in '+ country + ' (forecasting with holidays)')

                cols = cols_h

            # Save forecasting results 

            forecast_future_df_best = forecast_future_best[cols]

            forecast_future_df_best['country'] = country

            forecast_future_df_best.rename(columns={'yhat':'confirmed'}, inplace=True)

            if j == 0:                

                forecast_future_dfs = forecast_future_df_best.tail(days_to_forecast)

            else:

                forecast_future_dfs = pd.concat([forecast_future_dfs, forecast_future_df_best.tail(days_to_forecast)])

            

            # Save results to dataframe with result for the last date

            results.loc[j,'Country'] = country

            results.loc[j,'Country_code'] = country_iso_alpha

            confirmed_real_last = country_df_val.tail(1)['y'].values[0].astype('int')

            results.loc[j,'Conf_real'] = confirmed_real_last if confirmed_real_last > 0 else 0

            confirmed_pred_last = round(forecast_df.tail(1)['yhat'].values[0]).astype('int')

            results.loc[j,'Conf_pred'] = confirmed_pred_last if confirmed_pred_last > 0 else 0

            confirmed_pred_holidays_last = round(forecast_holidays_df_best.tail(1)['confirmed'].values[0],0).astype('int')

            results.loc[j,'Conf_pred_h'] = confirmed_pred_holidays_last if confirmed_pred_holidays_last > 0 else 0

            results.loc[j,'n_h'] = number_holidays

            results.loc[j,'err'] = relative_errors[-1]

            results.loc[j,'err_h'] = relative_errors_holidays[-1]

            results.loc[j,'lower_window'] = lower_window_best

            results.loc[j,'upper_window'] = upper_window_best

            results.loc[j,'prior_scale'] = prior_scale_best

            results.loc[j,'how_less, %'] = round((relative_errors[-1]-relative_errors_holidays[-1])*100/relative_errors[-1],1)

            model_future_best.plot_components(forecast_future_best)



        else:

            print('Country ' + str(country) + ' is not listed! ')

            continue

            

    return forecast_holidays_dfs, relative_errors_holidays, forecast_future_dfs, results
forecast_holidays_dfs, relative_errors_holidays, forecast_future_dfs, results = make_forecasts(all_countries, confirmed_df, holidays_df, 

                                                                                               days_to_forecast, days_to_forecast_for_evalution, first_forecasted_date)
forecast_future_dfs.head(3)
forecast_holidays_dfs.head(3)
pd.set_option('max_rows', 70)
print('Forecasting results')

display(results.sort_values(by=['err_h'], ascending=True))
df_h_impact = results[results['how_less, %'] > 1]

if len(df_h_impact) > 0:

    print('Countries with the impact of holidays')

    display(df_h_impact.sort_values(by=['how_less, %'], ascending=False))

    print('Number of these countries is', len(df_h_impact))
df_h_non_impact = results[results['how_less, %'] < -10]

if len(df_h_non_impact) > 0:

    print('Countries without the impact of holidays')

    display(df_h_non_impact.sort_values(by=['how_less, %'], ascending=False))

    print('Number of these countries is', len(df_h_non_impact))
df_neutral = results[(results['how_less, %'] <= 1) & (results['how_less, %'] >= -10)]

if len(df_neutral) > 0:

    print('Others countries')

    display(df_neutral.sort_values(by=['how_less, %'], ascending=False))

    print('Number of these countries is', len(df_neutral))
forecast_holidays_dfs.to_csv('forecast_holidays_dfs.csv', index=False)

forecast_future_dfs.to_csv('forecast_future_dfs.csv', index=False)

results.to_csv('results.csv', index=False)
results[['Country', 'Country_code', 'lower_window', 'upper_window', 'prior_scale']].to_csv('holidays_params.csv', index=False)