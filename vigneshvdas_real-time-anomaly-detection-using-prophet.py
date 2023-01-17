import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
# from influxdb import InfluxDBClient
# from influxdb import DataFrameClient
print("Imported Python Libraries")
# client = InfluxDBClient(host='********', port=****, username='****', password='*******', database='******')
print("Begining to create pandas DataFrame")
# The following is the query that we use to fetch data from InfluxDB
# q = "select query from InfluxDB"
# df = pd.DataFrame(client.query(q).get_points())

df = pd.read_csv('../input/influxblockeddatacsv/BlockedData.csv')

print("Created pandas DataFrame Successfully")

# Display the head of the data
df.head()
print("Beginning Data Preprocessing for Prophet")

# Rename the columns as prophet accepts columns only as 'ds' for date time and 'y' as value to forecast.
df.rename(columns={'Transactions_Blocked_Count':'y','time':'ds'}, inplace=True)

# Formatting the data received from InfluxDB
df['ds'] = df['ds'].str.replace('T',' ')
df['ds'] = df['ds'].str.replace('Z','')

# Remove any invalid non-numeric data from 'y'
df['y'] = df['y'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
df['y'] = df['y']

print("Data Processing is Done")
# Display the head of the data after processing
df.head()
# Build Prophet Model
m = Prophet()
m.fit(df)
print("Model Build is Successful")
# Predit using the model but forecasting is not applicable here.
forecast = m.predict()
print("Forecasting Done using the Model")
# Analyze weekly and daily trends
trend = m.plot_components(forecast)
df.tail()
# Plot the timeseries data
plotgraph = m.plot(forecast)
if df['y'][df.index[-1]] > forecast['yhat_upper'][forecast.index[-1]] and df['y'][df.index[-2]] > forecast['yhat_upper'][forecast.index[-2]]:
       print("Anomaly is detected as " + str(df['y'][df.index[-1]]) +  " is greater than forecast max " + str(forecast['yhat_upper'][forecast.index[-1]]))
else:
       print("No anomalies to report")