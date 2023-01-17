import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl

import plotly.offline as py

import folium

import Bio.SeqIO

init_notebook_mode(connected=True)

plt.rcParams.update({'font.size': 14})
confirmed_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

deaths_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

recoveries_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
# Drop date columns if they are mostly NaN

na_columns = (confirmed_df.isna().sum() / confirmed_df.shape[0]) > 0.99

na_columns = na_columns[na_columns]



confirmed_df = confirmed_df.drop(na_columns.index, axis=1)

deaths_df = deaths_df.drop(na_columns.index, axis=1)

recoveries_df = recoveries_df.drop(na_columns.index, axis=1)
## Tidy up the data

confirmed_df = confirmed_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='confirmed')

deaths_df = deaths_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='deaths')

recoveries_df = recoveries_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='recoveries')
confirmed_df['date'] = pd.to_datetime(confirmed_df['date'])

deaths_df['date'] = pd.to_datetime(deaths_df['date'])

recoveries_df['date'] = pd.to_datetime(recoveries_df['date'])
full_df = confirmed_df.merge(recoveries_df).merge(deaths_df)

full_df = full_df.rename(columns={'Country/Region': 'Country', 'date': 'Date', 'confirmed': "Confirmed", "recoveries": "Recoveries", "deaths": "Deaths"})

# Check null values

full_df.isnull().sum()
world_df = full_df.groupby(['Date']).agg({'Confirmed': ['sum'], 'Recoveries': ['sum'], 'Deaths': ['sum']}).reset_index()

world_df.columns = world_df.columns.get_level_values(0)



def add_rates(df):

    df['Confirmed Change'] = df['Confirmed'].diff().shift(-1)

 

    df['Mortality Rate'] = df['Deaths'] / df['Confirmed']

    df['Recovery Rate'] = df['Recoveries'] / df['Confirmed']

    df['Growth Rate'] = df['Confirmed Change'] / df['Confirmed']

    df['Growth Rate Change'] = df['Growth Rate'].diff().shift(-1)

    df['Growth Rate Accel'] = df['Growth Rate Change'] / df['Growth Rate']

    return df



world_df = add_rates(world_df)
def plot_aggregate_metrics(df, fig=None):

    if fig is None:

        fig = go.Figure()

    fig.update_layout(template='plotly_dark')

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Confirmed'],

                             mode='lines+markers',

                             name='Confirmed',

                             line=dict(color='Yellow', width=2)

                            ))

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Deaths'],

                             mode='lines+markers',

                             name='Deaths',

                             line=dict(color='Red', width=2)

                            ))

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Recoveries'],

                             mode='lines+markers',

                             name='Recoveries',

                             line=dict(color='Green', width=2)

                            ))

    return fig
plot_aggregate_metrics(world_df).show()
def plot_diff_metrics(df, fig=None):

    if fig is None:

        fig = go.Figure()



    fig.update_layout(template='plotly_dark')

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Mortality Rate'],

                             mode='lines+markers',

                             name='Mortality rate',

                             line=dict(color='red', width=2)))



    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Recovery Rate'],

                             mode='lines+markers',

                             name='Recovery rate',

                             line=dict(color='Green', width=2)))



    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Growth Rate'],

                             mode='lines+markers',

                             name='Growth rate confirmed',

                             line=dict(color='Yellow', width=2)))

    fig.update_layout(yaxis=dict(tickformat=".2%"))

    

    return fig
plot_diff_metrics(world_df).show()
fig = go.Figure()

fig.update_layout(template='plotly_dark')



tmp_df = world_df.copy()

tmp_df = tmp_df[tmp_df['Growth Rate Accel'] < 10]



fig.add_trace(go.Scatter(x=tmp_df['Date'], 

                         y=tmp_df['Growth Rate Accel'],

                         mode='lines+markers',

                         name='Growth Acceleration',

                         line=dict(color='Green', width=3)))

fig.update_layout(yaxis=dict(tickformat=".2%"))



fig.show()
confirmed_by_country_df = full_df.groupby(['Date', 'Country']).sum().reset_index()
fig = px.line(confirmed_by_country_df, x='Date', y='Confirmed', color='Country', line_group="Country", hover_name="Country")

fig.update_layout(template='plotly_dark')

fig.show()
# Log scale to allow for view

#  (1) of countries other than China, and

#  (2) identifying linear sections, which indicate exponential growth

fig = px.line(confirmed_by_country_df, x='Date', y='Confirmed', color='Country', line_group="Country", hover_name="Country")

fig.update_layout(

    template='plotly_dark',

    yaxis_type="log"

)

fig.show()
confirmed_by_country_df.groupby('Country').max().sort_values(by='Confirmed', ascending=False)[:60]
k_layout_kwargs = {

    'font': dict(size=12,),

    'legend': dict(x=0, y=-0.7),

}
bg_df = confirmed_by_country_df[confirmed_by_country_df['Country'] == 'Malta'].copy()

bg_df = add_rates(bg_df)
plot_aggregate_metrics(bg_df).show()
plot_diff_metrics(tmp_df).show()
from sklearn.linear_model import LinearRegression



bg_growth_rates = {}



n_days_to_fit = 7

confirmed_bg_df = confirmed_by_country_df[(confirmed_by_country_df['Country'] == 'Malta') & (confirmed_by_country_df['Date'] >= (np.datetime64('today') - np.timedelta64(n_days_to_fit,'D')))]



x = (confirmed_bg_df['Date'] - confirmed_bg_df['Date'].min()).dt.days.to_numpy().reshape(-1, 1)

y = np.log(confirmed_bg_df['Confirmed'])

reg = LinearRegression().fit(x, y)

print(f"Model fit score: {reg.score(x, y):.2f}")

print(f"Growth rate: {reg.coef_[0]:.3f}")

bg_growth_rates[n_days_to_fit] = reg.coef_[0]



fig = go.Figure()



fig.add_trace(

    go.Scatter(

        x=x[:,0],

        y=np.exp(y),

        name='Malta - Current fit'

    )

)



xx = np.linspace(0, len(x[:,0]) + 14, 100)  # Forecast 14 days out

yy = reg.predict(xx.reshape(-1,1))



fig.add_trace(

    go.Scatter(

        x=xx,

        y=np.exp(yy),

        name='Malta - Exponential fit',

        mode='lines',

    )

)



fig.update_layout(

    title=f"Exponential Model of Malta. Confirmed Cases<br>(fit to last {n_days_to_fit} days) with 14-Day Extrapolation",

    xaxis_title=f"Days since {confirmed_bg_df['Date'].min()}",

    yaxis_title="Number of Confirmed Cases",

    yaxis_type="log",

    template="plotly_dark",

    **k_layout_kwargs,

)



fig.show()



n_days_to_fit = 4

confirmed_bg_df = confirmed_by_country_df[(confirmed_by_country_df['Country'] == 'Malta') & (confirmed_by_country_df['Date'] >= (np.datetime64('today') - np.timedelta64(n_days_to_fit,'D')))]



x = (confirmed_bg_df['Date'] - confirmed_bg_df['Date'].min()).dt.days.to_numpy().reshape(-1, 1)

y = np.log(confirmed_bg_df['Confirmed'])



reg = LinearRegression().fit(x, y)

print(f"Model fit score: {reg.score(x, y):.2f}")

print(f"Growth rate: {reg.coef_[0]:.3f}")

bg_growth_rates[n_days_to_fit] = reg.coef_[0]



fig = go.Figure()



fig.add_trace(

    go.Scatter(

        x=confirmed_by_country_df[confirmed_by_country_df['Country'] == 'Malta']['Date'],

        y=confirmed_by_country_df[confirmed_by_country_df['Country'] == 'Malta']['Confirmed'],

        name='Malta - Current fit',

        line=dict(width=4)

    )

)



predict_days_out = 7*2



exponential_fit_date_range = pd.date_range(confirmed_bg_df['Date'].min(), confirmed_bg_df['Date'].max() + np.timedelta64(predict_days_out,'D'))



xx = np.linspace(0, len(x[:,0]) + predict_days_out, exponential_fit_date_range.shape[0])  # Forecast 14 days out

yy = reg.predict(xx.reshape(-1,1))



fig.add_trace(

    go.Scatter(

        x=exponential_fit_date_range,

        y=np.exp(yy),

        name='Malta - Exponential fit',

        mode='lines'

    )

)



fig.update_layout(

    title=f"Exponential Model of Malta Confirmed Cases<br>(fit to last {n_days_to_fit} days) with {predict_days_out}-Day Extrapolation",

    xaxis_title=f"Date",

    yaxis_title="Number of Confirmed Cases",

    yaxis_type="log",

    template="plotly_dark",

    **k_layout_kwargs,

)



fig.show()
proxy_country = 'Italy'



confirmed_italy_df = confirmed_by_country_df[(confirmed_by_country_df['Country'] == proxy_country) & (confirmed_by_country_df['Confirmed'] >= 100)]



x = (confirmed_italy_df['Date'] - confirmed_italy_df['Date'].min()).dt.days.to_numpy().reshape(-1, 1)

y = np.log(confirmed_italy_df['Confirmed'])

reg = LinearRegression().fit(x, y)

print(f"Model fit score: {reg.score(x, y):.2f}")

print(f"Growth rate: {reg.coef_[0]:.3f}")

italy_growth_rate = reg.coef_[0]



fig = go.Figure()



fig.add_trace(

    go.Scatter(

        x=x[:,0],

        y=np.exp(y),

        name=proxy_country

    )

)



xx = np.linspace(0, len(x[:,0]) + 14, 100)  # Forecast 14 days out

yy = reg.predict(xx.reshape(-1,1))



fig.add_trace(

    go.Scatter(

        x=xx,

        y=np.exp(yy),

        name=f'{proxy_country} - Exponential fit'

    )

)



fig.update_layout(

    title=f"Exponential Model of {proxy_country} Confirmed Cases<br>with 14-Day Extrapolation",

    xaxis_title=f"Days since {confirmed_italy_df['Date'].min()}",

    yaxis_title="Number of Confirmed Cases",

    yaxis_type="log",

    template="plotly_dark",

    **k_layout_kwargs,

)



fig.show()
def linear_model(x, a, b):

    return b * x + a





def linear_model_fixed_slope(slope):

    def func(x, intercept):

        return linear_model(x, a=intercept, b=slope)

    

    return func



test_model = linear_model_fixed_slope(2)

x = np.array([1, 2, 3])

test_model(x=x, intercept=2)
from scipy.optimize import curve_fit





def get_model(model, popt):

    def fitted_model(x):

        return model(x, *popt)

    return fitted_model





x = (confirmed_bg_df['Date'] - confirmed_bg_df['Date'].min()).dt.days.to_numpy()

y = np.log(confirmed_bg_df['Confirmed'].to_numpy())



# Pull the slope from the Italy model and use for the U.S., allowing only the intercept to vary

model = linear_model_fixed_slope(italy_growth_rate)

popt, pcov = curve_fit(model, x, y)



fitted_model_italy_rate = get_model(model, popt)



# Now do the same using the slope from the Malta model

bg_growth_rate_n_days_to_fit = 4

print(bg_growth_rate_n_days_to_fit)



model = linear_model_fixed_slope(bg_growth_rates[bg_growth_rate_n_days_to_fit])

popt, pcov = curve_fit(model, x, y)



fitted_model_bg_rate = get_model(model, popt)



# Plot results

for layout_kwargs in [{}, {"yaxis_type": "log"}]:

    fig = go.Figure()



    fig.add_trace(

        go.Scatter(

            x=confirmed_by_country_df[confirmed_by_country_df['Country'] == 'Malta']['Date'],

            y=confirmed_by_country_df[confirmed_by_country_df['Country'] == 'Malta']['Confirmed'],

            name='Malta - Current fit',

            line=dict(width=4)

        )

    )



    exponential_fit_date_range = pd.date_range(confirmed_bg_df['Date'].min(), confirmed_bg_df['Date'].max() + np.timedelta64(14,'D'))



    xx = np.linspace(0, len(x) + 14, exponential_fit_date_range.shape[0])  # Forecast 14 days out

    yy = fitted_model_italy_rate(xx)



    fig.add_trace(

        go.Scatter(

            x=exponential_fit_date_range,

            y=np.exp(yy),

            name=f'Malta - Exponential fit based on {proxy_country} growth rate ({italy_growth_rate:.0%})',

            mode='lines'

        )

    )



    #########



    yy = fitted_model_bg_rate(xx)



    fig.add_trace(

        go.Scatter(

            x=exponential_fit_date_range,

            y=np.exp(yy),

            name=f'Malta - Exponential fit based on US growth rate ({bg_growth_rates[bg_growth_rate_n_days_to_fit]:.0%}) (fitted to {bg_growth_rate_n_days_to_fit} days)',

            mode='lines'

        )

    )



    fig.update_layout(

        title="Exponential Model of Malta Confirmed Cases<br>with 14-Day Extrapolation",

        xaxis_title="Date",

        yaxis_title="Number of Confirmed Cases",

        template="plotly_dark",

        **k_layout_kwargs,

        **layout_kwargs

    )



    fig.show()
from fbprophet.plot import plot_plotly

from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot
full_pop = 330e6



#floor_model = lambda x: max(x - 1, 0)  # Use the value itself because the function only increases

floor_model = lambda x: round(0.65 * x)

cap_model = lambda x: round(min(full_pop, 1.5 * x + 10000))  # 50% above plus one to ensure floor > cap at 0



# Modeling Malta confirmed cases 

confirmed_training_df = confirmed_by_country_df[(confirmed_by_country_df['Country'] == 'Malta') & (confirmed_by_country_df['Confirmed'] > 0)]

confirmed_training_df = confirmed_training_df.rename(columns={'Date': 'ds', 'Confirmed': 'y'}).reset_index(drop=True)



confirmed_training_df['floor'] = confirmed_training_df.y.apply(floor_model)

confirmed_training_df['cap'] = confirmed_training_df.y.apply(cap_model)
confirmed_training_df.y = confirmed_training_df.y.apply(np.log10)

confirmed_training_df.floor = confirmed_training_df.floor.apply(np.log10)

confirmed_training_df.cap = confirmed_training_df.cap.apply(np.log10)
# Total confirmed model 

m = Prophet(

    growth='linear',

    #interval_width=0.90,

    changepoint_prior_scale=0.05,

    changepoint_range=0.9,

    yearly_seasonality=False,

    weekly_seasonality=False,

    daily_seasonality=False,

    #n_changepoints=2

)

m.fit(confirmed_training_df)

future = m.make_future_dataframe(periods=14)

future['floor'] = confirmed_training_df.floor

future['cap'] = confirmed_training_df.cap

confirmed_forecast = m.predict(future)
for kwargs in [{}, {"yaxis_type": "log"}]:

    fig = plot_plotly(m, confirmed_forecast, plot_cap=False, changepoints=True)

    annotations = []

    annotations.append(dict(

        xref='paper',

        yref='paper',

        x=0.0,

        y=1.15,

        xanchor='left',

        yanchor='bottom',

        text='Predictions for log10 Confirmed cases Malta',

        font=dict(

            family='Arial',

            size=30,

            color='rgb(37,37,37)'),

        showarrow=False))

    fig.update_layout(

        annotations=annotations,

        **kwargs

    )

    fig.show()
for kwargs in [{}, {"yaxis_type": "log"}]:

    fig = plot_plotly(m, confirmed_forecast, plot_cap=False, changepoints=True)

    annotations = []

    annotations.append(dict(

        xref='paper',

        yref='paper',

        x=0.0,

        y=1.15,

        xanchor='left',

        yanchor='bottom',

        text='Predictions for Confirmed cases Malta',

        font=dict(

            family='Arial',

            size=30,

            color='rgb(37,37,37)'),

        showarrow=False))

    fig.update_layout(

        annotations=annotations,

        **kwargs

    )

    for trace in fig.data:

        trace.y = np.power(trace.y, 10)

    fig.show()