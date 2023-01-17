import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import plotly.graph_objects as go
import warnings
import datetime
import math
from scipy.optimize import minimize

# Configure Jupyter Notebook
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 500) 
pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)
display(HTML("<style>div.output_scroll { height: 35em; }</style>"))

%matplotlib inline
%config InlineBackend.figure_format ='retina'

warnings.filterwarnings('ignore')
# the number of days into the future for the forecast
days_forecast = 40
# download the latest data sets
conf_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
recv_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
# create full table
dates = conf_df.columns[4:]

conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Confirmed')

deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Deaths')

recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Recovered')

full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'], recv_df_long['Recovered']], 
                       axis=1, sort=False)

# avoid double counting
full_table = full_table[full_table['Province/State'].str.contains(',')!=True]
# cases 
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Active Case = confirmed - deaths - recovered
full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

# replacing Mainland china with just China
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

# filling missing values 
full_table[['Province/State']] = full_table[['Province/State']].fillna('')
# Display the number cases globally
df = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
df = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df =  df[df['Date']==max(df['Date'])].reset_index(drop=True)
df
# start a dataframe with the unique countries and provinces. 
# This will be used to keep track of the model performance for each one
df_model_select = full_table.groupby(['Country/Region','Province/State']).last().reset_index()
df_model_select = df_model_select[['Country/Region','Province/State','Confirmed']] 
df_model_select['Offset error'] = 100
df_model_select['No Offset error'] = 100
countries = list(set(full_table['Country/Region']))
countries.sort()

for country in countries:
    clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
    clusters.sort()
    
    for cluster in clusters:
        print(' ')
        print('-----------------')
        print(country + ' - ' + cluster)
        
        df = full_table[(full_table['Country/Region'] == country)&(full_table['Province/State'] == cluster)]
        df = df.groupby(['Date','Country/Region']).sum().reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date'])
        df = df.set_index('Date')[['Confirmed']]
        df.drop(df.tail(3).index,inplace=True)
        df_result = df.copy()
        # df_result = df_result[['Date','Confirmed']]

        # ensure that the model starts from when the first case is detected
        # NOTE: its better not to truncate the dataset like this 
        # df = df[df[df.columns[0]]>0]

        # define the models to forecast the growth of cases
        def model(N, a, alpha, t0, t):
            return N * (1 - math.e ** (-a * (t-t0))) ** alpha

        def model_loss(params):
            N, a, alpha, t0 = params
            global df
            r = 0
            for t in range(len(df)):
                r += (model(N, a, alpha, t0, t) - df.iloc[t, 0]) ** 2
            return r 
        try:
            N = df['Confirmed'][-1]
            T = -df['Confirmed'][0]
        except:
            N = 10000
            T = 0

        opt = minimize(model_loss, x0=np.array([N, 1.5, 5, T]), method='Nelder-Mead', tol=1e-7).x
        print('Offset' + str(opt))

        # create series to be plotted 
        x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
        x_actual =list(x_actual)
        y_actual = list(df.reset_index().iloc[:,1])

        start_date = pd.to_datetime(df.index[0])

        x_model = []
        y_model = []

        # get the model values for the same time series as the actuals
        for t in range(len(df) + days_forecast):
            x_model.append(start_date + datetime.timedelta(days=t))
            y_model.append(round(model(*opt,t)))

        # now add the results of the model to the dataframe
        df2 = pd.DataFrame(y_model,index=x_model,columns=['Offset'])
        df2.index.name = 'Date'
        df_result = pd.merge(df_result,
                             df2,
                             how='outer',
                             left_on=['Date'],
                             right_on=['Date'])

        # define the models to forecast the growth of cases
        def model(N, a, alpha, t):
            return N * (1 - math.e ** (-a * (t))) ** alpha

        def model_loss(params):
            N, a, alpha = params
            global df
            r = 0
            # error minimization should prefer (weigh more on) the larger population in this case, therefore no normalization.
            for t in range(len(df)):
                r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
            return r 

        try:
            N = df['Confirmed'][-1]
        except:
            N = 10000

        opt = minimize(model_loss, x0=np.array([N, 1.5, 5]), method='Nelder-Mead', tol=1e-7).x
        print('No offset' + str(opt))

        try:
            start_date = pd.to_datetime(df.index[0])

            x_model = []
            y_model = []

            # get the model values for the same time series as the actuals
            for t in range(len(df) + days_forecast):
                x_model.append(start_date + datetime.timedelta(days=t))
                y_model.append(round(model(*opt,t)))

            # now add the results of the model to the dataframe
            df2 = pd.DataFrame(y_model,index=x_model,columns=['No Offset'])
            df2.index.name = 'Date'
            df_result = pd.merge(df_result,
                                 df2,
                                 how='outer',
                                 left_on=['Date'],
                                 right_on=['Date'])
            
            df_result = df_result[df_result['Confirmed'].notnull()]
            err_offset = 0
            err_no_offset = 0
            for t in range(len(df_result)):
                err_offset += (math.log(df_result['Offset'].iloc[t]+1)-math.log(df_result['Confirmed'].iloc[t]+1))**2
                err_no_offset += (math.log(df_result['No Offset'].iloc[t]+1)-math.log(df_result['Confirmed'].iloc[t]+1))**2

            err_offset = math.sqrt(err_offset/len(df_result))
            err_no_offset = math.sqrt(err_no_offset/len(df_result))
            
            df_model_select['Offset error'][(df_model_select['Country/Region']==country)&(df_model_select['Province/State']==cluster)] = err_offset
            df_model_select['No Offset error'][(df_model_select['Country/Region']==country)&(df_model_select['Province/State']==cluster)] = err_no_offset
        except:
            pass
def highlight_max(s):
    '''
    highlight the absolute maximum value in a Series with red font.
    '''
    is_min = abs(s) == abs(s).max()
    return ['color: red' if v else '' for v in is_min]

df_model_select.style.apply(highlight_max,axis=1,subset=['Offset error', 'No Offset error'])
# plot the errors
x_model = list(df_model_select['Country/Region']+' - '+ df_model_select['Province/State'])
y_model_off = list(df_model_select['Offset error'])
y_model_noff = list(df_model_select['No Offset error'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title='Error comparison',
                  xaxis_title='Date',
                  yaxis_title="error",
                  autosize=False,
                  width=750,
                  height=800,
                  #yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_off,
                      mode='lines',
                      name='Offset error',
                      line=dict(color='blue', 
                                width=1.5
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_noff,
                      mode='lines',
                      name='No Offset error',
                      line=dict(color='red', 
                                width=1.0,
                                dash='dot'
                               )
                     ) 
             )

fig.show()
# start a dataframe with the unique countries and provinces. 
# This will be used to keep track of the model performance for each one
df_model_select = full_table.groupby(['Country/Region','Province/State']).last().reset_index()
df_model_select = df_model_select[['Country/Region','Province/State','Deaths']] 
df_model_select['Offset error'] = 100
df_model_select['No Offset error'] = 100
countries = list(set(full_table['Country/Region']))
countries.sort()

for country in countries:
    clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
    clusters.sort()
    
    for cluster in clusters:
        print(' ')
        print('-----------------')
        print(country + ' - ' + cluster)
        
        df = full_table[(full_table['Country/Region'] == country)&(full_table['Province/State'] == cluster)]
        df = df.groupby(['Date','Country/Region']).sum().reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date'])
        df = df.set_index('Date')[['Deaths']]
        df.drop(df.tail(3).index,inplace=True)
        df_result = df.copy()
        # df_result = df_result[['Date','Deaths']]

        # ensure that the model starts from when the first case is detected
        # NOTE: its better not to truncate the dataset like this 
        # df = df[df[df.columns[0]]>0]

        # define the models to forecast the growth of cases
        def model(N, a, alpha, t0, t):
            return N * (1 - math.e ** (-a * (t-t0))) ** alpha

        def model_loss(params):
            N, a, alpha, t0 = params
            global df
            r = 0
            for t in range(len(df)):
                r += (model(N, a, alpha, t0, t) - df.iloc[t, 0]) ** 2
            return r 
        try:
            N = df['Deaths'][-1]
            T = -df['Deaths'][0]
        except:
            N = 10000
            T = 0

        opt = minimize(model_loss, x0=np.array([N, 1.5, 5, T]), method='Nelder-Mead', tol=1e-7).x
        print('Offset' + str(opt))

        # create series to be plotted 
        x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
        x_actual =list(x_actual)
        y_actual = list(df.reset_index().iloc[:,1])

        start_date = pd.to_datetime(df.index[0])

        x_model = []
        y_model = []

        # get the model values for the same time series as the actuals
        for t in range(len(df) + days_forecast):
            x_model.append(start_date + datetime.timedelta(days=t))
            y_model.append(round(model(*opt,t)))

        # now add the results of the model to the dataframe
        df2 = pd.DataFrame(y_model,index=x_model,columns=['Offset'])
        df2.index.name = 'Date'
        df_result = pd.merge(df_result,
                             df2,
                             how='outer',
                             left_on=['Date'],
                             right_on=['Date'])

        # define the models to forecast the growth of cases
        def model(N, a, alpha, t):
            return N * (1 - math.e ** (-a * (t))) ** alpha

        def model_loss(params):
            N, a, alpha = params
            global df
            r = 0
            # error minimization should prefer (weigh more on) the larger population in this case, therefore no normalization.
            for t in range(len(df)):
                r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
            return r 

        try:
            N = df['Deaths'][-1]
        except:
            N = 10000

        opt = minimize(model_loss, x0=np.array([N, 1.5, 5]), method='Nelder-Mead', tol=1e-7).x
        print('No offset' + str(opt))

        try:
            start_date = pd.to_datetime(df.index[0])

            x_model = []
            y_model = []

            # get the model values for the same time series as the actuals
            for t in range(len(df) + days_forecast):
                x_model.append(start_date + datetime.timedelta(days=t))
                y_model.append(round(model(*opt,t)))

            # now add the results of the model to the dataframe
            df2 = pd.DataFrame(y_model,index=x_model,columns=['No Offset'])
            df2.index.name = 'Date'
            df_result = pd.merge(df_result,
                                 df2,
                                 how='outer',
                                 left_on=['Date'],
                                 right_on=['Date'])
            
            df_result = df_result[df_result['Deaths'].notnull()]
            err_offset = 0
            err_no_offset = 0
            for t in range(len(df_result)):
                err_offset += (math.log(df_result['Offset'].iloc[t]+1)-math.log(df_result['Deaths'].iloc[t]+1))**2
                err_no_offset += (math.log(df_result['No Offset'].iloc[t]+1)-math.log(df_result['Deaths'].iloc[t]+1))**2

            err_offset = math.sqrt(err_offset/len(df_result))
            err_no_offset = math.sqrt(err_no_offset/len(df_result))
            
            df_model_select['Offset error'][(df_model_select['Country/Region']==country)&(df_model_select['Province/State']==cluster)] = err_offset
            df_model_select['No Offset error'][(df_model_select['Country/Region']==country)&(df_model_select['Province/State']==cluster)] = err_no_offset
        except:
            pass
def highlight_max(s):
    '''
    highlight the absolute maximum value in a Series with red font.
    '''
    is_min = abs(s) == abs(s).max()
    return ['color: red' if v else '' for v in is_min]

df_model_select.style.apply(highlight_max,axis=1,subset=['Offset error', 'No Offset error'])
# plot the errors
x_model = list(df_model_select['Country/Region']+' - '+ df_model_select['Province/State'])
y_model_off = list(df_model_select['Offset error'])
y_model_noff = list(df_model_select['No Offset error'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title='Error comparison',
                  xaxis_title='Date',
                  yaxis_title="% error",
                  autosize=False,
                  width=750,
                  height=800,
                  #yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_off,
                      mode='lines',
                      name='Offset error',
                      line=dict(color='blue', 
                                width=1.5
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_noff,
                      mode='lines',
                      name='No Offset error',
                      line=dict(color='red', 
                                width=1.0,
                                dash='dot'
                               )
                     ) 
             )

fig.show()
df_ca_train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
df_ca_test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
df_ca_submission = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')
df_ca_train.tail(10)
full_table = df_ca_train
full_table[['Province/State']] = full_table[['Province/State']].fillna('')
df_comp = df_ca_test
df_comp[['Province/State']] = df_comp[['Province/State']].fillna('')
df_comp['Date'] = pd.to_datetime(df_comp['Date'])
df_comp['ConfirmedCases']=0
df_comp['Fatalities']=0
countries = list(set(full_table['Country/Region']))
countries.sort()

for country in countries:
    clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
    clusters.sort()
    
    for cluster in clusters:
        print(' ')
        print('-----------------')
        print(str(country) + ' - ' + str(cluster))
        
        df = full_table[(full_table['Country/Region'] == country)&(full_table['Province/State'] == cluster)]
        df = df.groupby(['Date','Country/Region']).sum().reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date'])
        df = df.set_index('Date')[['ConfirmedCases']]

        df_result = df.copy()
        # df_result = df_result[['Date','Confirmed']]

        # define the models to forecast the growth of cases
        def model(N, a, alpha, t):
            return N * (1 - math.e ** (-a * (t))) ** alpha

        def model_loss(params):
            N, a, alpha = params
            global df
            r = 0
            for t in range(len(df)):
                r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
            return r 
        try:
            N = df['ConfirmedCases'][-1]
        except:
            N = 10000

        opt = minimize(model_loss, x0=np.array([N, 1.5, 5]), method='Nelder-Mead', tol=1e-7).x
        print(opt)
        
        x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
        x_actual = list(x_actual)
        y_actual = list(df.reset_index().iloc[:,1])
        
        start_date = pd.to_datetime(df.index[0])
        days_forecast = len(df)+len(df_ca_test)-7

        x_model = []
        y_model = []

        for t in range(days_forecast):
            x_model.append(start_date + datetime.timedelta(days=t))
            y_model.append(round(model(*opt,t)))
        
        # now add the results of the model to the competition dataframe
        df2 = pd.DataFrame(y_model,index=x_model,columns=['ConfirmedCases'])
        df2.index.name = 'Date'
        df2['Country/Region']=country
        df2['Province/State']=cluster
        df_comp = pd.merge(df_comp,
                             df2,
                             how='left',
                             on=['Date','Country/Region','Province/State']
                          )
        
        df_comp = df_comp.rename(columns={'ConfirmedCases_y': 'ConfirmedCases'})
        df_comp['ConfirmedCases'] = df_comp['ConfirmedCases'].fillna(df_comp['ConfirmedCases_x'])
        df_comp = df_comp[['ForecastId','Province/State','Country/Region','Date','ConfirmedCases','Fatalities']]

df_comp.head()

countries = list(set(full_table['Country/Region']))
countries.sort()

for country in countries:
    clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
    clusters.sort()
    
    for cluster in clusters:
        print(' ')
        print('-----------------')
        print(str(country) + ' - ' + str(cluster))
        
        df = full_table[(full_table['Country/Region'] == country)&(full_table['Province/State'] == cluster)]
        df = df.groupby(['Date','Country/Region']).sum().reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date'])
        df = df.set_index('Date')[['Fatalities']]

        df_result = df.copy()

        # define the models to forecast the growth of cases
        def model(N, a, alpha, t):
            return N * (1 - math.e ** (-a * (t))) ** alpha

        def model_loss(params):
            N, a, alpha = params
            global df
            r = 0
            for t in range(len(df)):
                r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
            return r 
        try:
            N = df['Fatalities'][-1]
        except:
            N = 10000

        opt = minimize(model_loss, x0=np.array([N, 1.5, 5]), method='Nelder-Mead', tol=1e-7).x
        print(opt)
        
        x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
        x_actual = list(x_actual)
        y_actual = list(df.reset_index().iloc[:,1])
        
        start_date = pd.to_datetime(df.index[0])
        days_forecast = len(df)+len(df_ca_test)-7

        x_model = []
        y_model = []

        for t in range(days_forecast):
            x_model.append(start_date + datetime.timedelta(days=t))
            y_model.append(round(model(*opt,t)))
        
        # now add the results of the model to the competition dataframe
        df2 = pd.DataFrame(y_model,index=x_model,columns=['Fatalities'])
        df2.index.name = 'Date'
        df2['Country/Region']=country
        df2['Province/State']=cluster
        df_comp = pd.merge(df_comp,
                             df2,
                             how='left',
                             on=['Date','Country/Region','Province/State']
                          )
        
        df_comp = df_comp.rename(columns={'Fatalities_y': 'Fatalities'})
        df_comp['Fatalities'] = df_comp['Fatalities'].fillna(df_comp['Fatalities_x'])
        df_comp = df_comp[['ForecastId','Province/State','Country/Region','Date','ConfirmedCases','Fatalities']]

df_comp[df_comp['Fatalities']>0]
df_comp.head()
df_ca_test.head()
df_ca_test['Date'] = pd.to_datetime(df_ca_test['Date'])
df_ca_submission.info()
df_sub = df_comp[['ForecastId','ConfirmedCases','Fatalities']]
df_sub.head()
# Writing the csv file
df_sub.to_csv('submission.csv',index=False)