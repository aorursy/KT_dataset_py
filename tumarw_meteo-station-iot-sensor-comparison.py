import datetime
import numpy as np
import os
import pandas as pd
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py

init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.

resampling_rate = '1H'

print(os.listdir("../input"))
# Load Meteo weather station data
meteo_df = pd.read_excel(open(os.path.join('../input','Rwinkwavu.xlsx'),'rb'))
meteo_df['Date'] = meteo_df['Date'].astype(str)

# Replace '.' with ':'
num_rows = meteo_df.shape[0]
meteo_df['Hour'] = meteo_df['Hour'].str.replace('.', ':')

# Combine Date and Hour columns
meteo_df['Date'] = pd.to_datetime(meteo_df['Date'] +  ' ' + meteo_df['Hour'].replace('.', ':'))
meteo_df.head(n=5)
# Format Meteo DataFrame
meteo_df.set_index('Date', inplace=True)
meteo_df.index.rename('date', inplace=True)
meteo_df.drop(labels=['Hour','(13-RWINKWAVU 2M AWS) (1) Temperature Minimun - °C', '(13-RWINKWAVU 2M AWS) (1) Temperature Maximun - °C'], axis=1, inplace=True)
meteo_df.rename(columns={"(13-RWINKWAVU 2M AWS) (1) Temperature Average - °C": "meteo-temp"}, inplace=True)

meteo_df = meteo_df.resample(resampling_rate).mean()

meteo_df.head(n=5)
# Load IoT sensor data
iot_df = pd.read_csv(open(os.path.join('../input', 'Rwinkwavu03.csv'),'rb'))
iot_df.set_index('date', inplace=True)
iot_df.rename(index=str, columns={'s1': 'iot-temp'}, inplace=True)
iot_df.index = pd.to_datetime(iot_df.index)
iot_df.drop(labels=['s2', 's3', 's4', 's5', 's6', 's7', 's8'], axis=1, inplace=True)
iot_df.head(n=5)

iot_df = iot_df.resample(resampling_rate).mean()
iot_df.head(n=5)
# Combine dataframes

temp_df = meteo_df.merge(iot_df, how='left', left_index=True, right_index=True)
temp_df.index = pd.to_datetime(temp_df.index)

temp_df = temp_df.resample('1H').mean()


temp_df = temp_df[temp_df['iot-temp'] > 0]
temp_df.head(n=5)
# Plot Meteo vs. IoT sensor temperature
def plot_temperature(temp_df):

    random_x = np.linspace(0, 1, num_rows)
        
    meteo_temp = go.Scatter(
        x = temp_df.index,
        y = temp_df['meteo-temp'],
        mode = 'lines',
        name = "Meteo Temp",
        line = dict(
            color = ('rgb(50, 150, 50)'),
            width = 2
        )
    )

    iot_temp = go.Scatter(
        x = temp_df.index,
        y = temp_df['iot-temp'],
        mode = 'lines',
        name = "IoT Sensor Temp",
        line = dict(
            color = ('rgb(50, 50, 180)'),
            width = 2
        ))


    data = [
        meteo_temp,
        iot_temp
    ]

    iplot(data, filename='meteo-vs-iot-temp.html')

plot_temperature(temp_df)
