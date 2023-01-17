import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go



from datetime import date, timedelta

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
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)
df["Country"].unique()
df_confirmed["Country"].unique()
np.array(set(df["Country"].unique()).difference(set(df_confirmed["Country"].unique())))
# Convert name of countries to ISO 3166 and equivalence of country names in dataframes df and df_confirmed

df["Country"].replace({'UK': 'United Kingdom'}, inplace=True)

df["Country"].replace({'US': 'United States'}, inplace=True)

df["Country"].replace({'Russia': 'Russian Federation'}, inplace=True)

df["Country"].replace({'South Korea': 'Korea, Republic of'}, inplace=True)

df["Country"].replace({'Mainland China': 'China'}, inplace=True)

df["Country"].replace({'Czech Republic': 'Czechia'}, inplace=True)

df_confirmed["Country"].replace({'UK': 'United Kingdom'}, inplace=True)

df_confirmed["Country"].replace({'US': 'United States'}, inplace=True)

df_confirmed["Country"].replace({'Russia': 'Russian Federation'}, inplace=True)

df_confirmed["Country"].replace({'Korea, South': 'Korea, Republic of'}, inplace=True)
df_countries = df['Country'].unique()

df_countries
df_confirmed.head()
df
df_confirmed = df_confirmed[["Province/State","Lat","Long","Country"]]

df_temp = df.copy()

df_latlong = pd.merge(df_temp, df_confirmed, on=["Country", "Province/State"])
df2 = df.groupby(["Date", "Country"])[['Date', 'Country', 'Confirmed']].sum().reset_index()

confirmed = df2.groupby(['Date', 'Country']).sum()[['Confirmed']].reset_index()
latest_date = confirmed['Date'].max()

latest_date
confirmed = confirmed[(confirmed['Date']==latest_date)][['Country', 'Confirmed']]
confirmed2 = confirmed.copy()
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
df2 = adaption_df_to_holidays_df_for_prophet(df2, 'Country', countries_dict)

df2
df2.columns
print("Number of countries/regions with data: " + str(len(df2.Country.unique())))
df2.describe()
df2.head()
df2.tail()
prior_scale_list = [0.05, 0.5, 1, 2, 5, 10, 15, 20, 40]
days_to_forecast = 14 # changable

first_forecasted_date = sorted(list(set(df2['Date'].values)))[-days_to_forecast]



print('The first date to perform forecasts for is: ' + str(first_forecasted_date))
confirmed_df = df2[['Date', 'Country', 'Confirmed', 'iso_alpha']]

confirmed_df
all_countries = confirmed_df['Country'].unique()

all_countries
def make_forecasts(all_countries, confirmed_df, holidays_df, days_to_forecast, first_forecasted_date):

    

    def eval_error(forecast_df, country_df_val, first_forecasted_date, title):

        # Evaluate forecasts with validation set val_df and calculaction and printing with title the relative error

        forecast_df[forecast_df['yhat'] < 0]['yhat'] = 0

        result_df = forecast_df[(forecast_df['ds'] >= pd.to_datetime(first_forecasted_date))]

        result_val_df = result_df.merge(country_df_val, on=['ds'])

        result_val_df['rel_diff'] = (result_val_df['y'] - result_val_df['yhat']).abs()

        relative_error = [sum(result_val_df['rel_diff'].values)*100/result_val_df['y'].sum()]

        

        # Check the output

        print(f'Result_val_df {title}:')

        print(relative_error[0], "% \n")

        

        return relative_error

    

    def model_training_forecasting(holidays_df=None):

        # Prophet model training and forecasting

        

        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False, 

                        holidays=holidays_df, changepoint_range=1, changepoint_prior_scale = 0.1,

                        seasonality_mode = 'multiplicative')

        model.add_seasonality(name='weekly', period=7, fourier_order=10, mode = 'additive')

        model.fit(country_df)

        future = model.make_future_dataframe(periods=days_to_forecast)

        forecast = model.predict(future)

        forecast[forecast['yhat'] < 0]['yhat'] = 0

        return model, forecast



    

    forecast_dfs = []

    relative_errors = []

    forecast_holidays_dfs = []

    relative_errors_holidays = []

    counter = 0

    results = pd.DataFrame(columns=['Country', 'Conf_real', 'Conf_pred', 'Conf_pred_h', 'n_h', 'err', 'err_h', 'prior_scale', 'how_less, %'])

    

    for j in range(len(all_countries)):

        country = all_countries[j]

        if country in confirmed_df['Country'].values:

            print(f'Country {str(country)} is listed')

            country_df = confirmed_df[confirmed_df['Country'] == country].reset_index(drop=True)

            country_iso_alpha = country_df['iso_alpha'][0]

            country_holidays_df = holidays_df[holidays_df['code'] == country_iso_alpha][['ds', 'holiday', 'lower_window', 'upper_window', 'prior_scale']].reset_index(drop=True)

            country_dfs = []            

            

            # Data preparation for forecast with Prophet

            country_df = country_df[['Date', 'Confirmed']]

            country_df.columns = ['ds','y']

            country_df['ds'] = pd.to_datetime(country_df['ds'])



            # Set training and validation datasets

            country_df_val = country_df[(country_df['ds'] >= pd.to_datetime(first_forecasted_date))]

            country_df = country_df[(country_df['ds'] < pd.to_datetime(first_forecasted_date))]



            # Without holidays

            # Model training and forecasting without holidays

            model, forecast = model_training_forecasting()

            #fig1 = model.plot_components(forecast)



            # Evaluate forecasts with validation set val_df and calculaction and printing the relative error

            forecast_df = forecast[['ds', 'yhat']].copy()

            relative_errors += eval_error(forecast_df, country_df_val, first_forecasted_date, 'without holidays')



            # With holidays

            # Model training with tuning prior_scale and forecasting

            relative_error_holidays_min = relative_errors[-1]

            for i in range(len(prior_scale_list)):

                country_holidays_df['prior_scale'] = prior_scale_list[i]

                number_holidays = len(country_holidays_df[(country_holidays_df['ds'] > '2020-01-21') & (country_holidays_df['ds'] < '2020-07-21')])

                model_holidays, forecast_holidays = model_training_forecasting(country_holidays_df)



                # Evaluate forecasts with validation set val_df and calculaction and printing the relative error

                forecast_holidays_df = forecast_holidays[['ds', 'yhat']].copy()

                relative_error_holidays = eval_error(forecast_holidays_df, country_df_val, first_forecasted_date, 'with holidays impact')

                

                # Save results

                if i == 0:

                    relative_error_holidays_min = relative_error_holidays

                    forecast_holidays_df_best = forecast_holidays

                    model_holidays_best = model_holidays

                    prior_scale_best = prior_scale_list[0]

                    

                elif (relative_error_holidays[0] < relative_error_holidays_min[0]):

                    relative_error_holidays_min = relative_error_holidays

                    forecast_holidays_df_best = forecast_holidays

                    model_holidays_best = model_holidays

                    prior_scale_best = prior_scale_list[i]

                    

                print('prior_scale =', prior_scale_list[i], 'relative_error_holidays_min',relative_error_holidays_min[0])

            

            # Results visualization

            print('The best errors of model with holidays is', relative_error_holidays_min[0], 'with prior_scale', str(prior_scale_best))

            relative_errors_holidays += relative_error_holidays_min

            

            if relative_errors[-1] < relative_errors_holidays[-1]:

                # The forecast without taking into account the holidays is the best

                forecast_plot = model.plot(forecast_holidays_df_best, ylabel='Confirmed in '+ country + ' (forecasting without holidays)')

            else:

                # The forecast taking into account the holidays is the best

                forecast_plot = model_holidays_best.plot(forecast_holidays_df_best, ylabel='Confirmed in '+ country + ' (forecasting with holidays)')            



            # Save results to dataframe with all dates

            forecast_holidays_df_best['Country'] = country

            forecast_holidays_df_best.rename(columns={'yhat':'Confirmed'}, inplace=True)

            forecast_holidays_dfs += [forecast_holidays_df_best.tail(days_to_forecast)]



            # Save results to dataframe with result for the last date

            results.loc[j,'Country'] = country

            confirmed_real_last = country_df_val.tail(1)['y'].values[0].astype('int')

            results.loc[j,'Conf_real'] = confirmed_real_last if confirmed_real_last > 0 else 0

            confirmed_pred_last = round(forecast_df.tail(1)['yhat'].values[0]).astype('int')

            results.loc[j,'Conf_pred'] = confirmed_pred_last if confirmed_pred_last > 0 else 0

            confirmed_pred_holidays_last = round(forecast_holidays_df_best.tail(1)['Confirmed'].values[0],0).astype('int')

            results.loc[j,'Conf_pred_h'] = confirmed_pred_holidays_last if confirmed_pred_holidays_last > 0 else 0

            results.loc[j,'n_h'] = number_holidays

            results.loc[j,'err'] = relative_errors[-1]

            results.loc[j,'err_h'] = relative_errors_holidays[-1]

            results.loc[j,'prior_scale'] = prior_scale_best

            results.loc[j,'how_less, %'] = round((relative_errors[-1]-relative_errors_holidays[-1])*100/relative_errors[-1],1)

            if round((relative_errors[-1]-relative_errors_holidays[-1])*100/relative_errors[-1],1) > 1:

                model_holidays_best.plot_components(forecast_holidays_df_best)

                

                if relative_errors_holidays[-1] < 1:

                    # Diagnostics by cross-validation

                    forecast_holidays_dfs_cv = cross_validation(model_holidays_best, initial= str(len(country_df)-28) +' days', period='7 days', horizon = '14 days')

                    forecast_holidays_dfs_cv_diagn = performance_metrics(forecast_holidays_dfs_cv, metrics=['mape'], rolling_window=1)

                    forecast_holidays_dfs_cv_diagn.rename(columns={'mape':'MAPE, %'}, inplace=True)

                    display(forecast_holidays_dfs_cv_diagn.head())

                    fig = plot_cross_validation_metric(forecast_holidays_dfs_cv, metric='mape')



        else:

            print('Country ' + str(country) + ' is not listed! ')

            continue

            

    return forecast_holidays_dfs, relative_errors_holidays, results
forecast_holidays_dfs, relative_errors_holidays, results = make_forecasts(all_countries, confirmed_df, holidays_df, days_to_forecast, first_forecasted_date)
print('Countries with the significant impact of holidays')

display(results[results['how_less, %'] > 1].sort_values(by=['how_less, %'], ascending=False))
print('Number of these countries is', len(results[results['how_less, %'] > 1]))
print('Countries without the significant impact of holidays')

display(results[results['how_less, %'] <= 1].sort_values(by=['how_less, %'], ascending=False))
print('Number of these countries is', len(results[results['how_less, %'] <= 1]))