import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv', low_memory = False)
data.head(5)
print('No. of Regions: %i' %data['Region'].nunique())

print('No. of Countries: %i' %data['Country'].nunique())

print('No. of States: %i' %data['State'].nunique())

print('No. of Cities: %i' %data['City'].nunique())
print(data["Region"].unique())

print(data["Country"].unique())

print(data["Month"].unique())

print(data["Day"].unique())

print(data["Year"].unique())
data.info()
data.describe()
data.isna().sum()
# Removing '201' and '200' from Year column

df = data[~data['Year'].isin(['201','200'])]



# Removing '0' from Date column

df = df[df['Day'] != 0]



# Removing 'State' column

df = df.drop(columns=['State'])



# Dropping the rowns with temperature -99

df = df.drop(df[df['AvgTemperature'] == -99.0].index)
# Adding a row with average temperatures in Celcius

df['AvgTempCelcius'] = round((((df.iloc[:,6] - 32) * 5) / 9),2)



# Adding the Date column in the format YYYY-MM-DD

df['Date'] = df.iloc[:,5].astype(str) + '-' + df.iloc[:,3].astype(str) + '-' + df.iloc[:,4].astype(str)



# Coverting the Date column into Pandas Date type datetime64[ns]

df['Date'] = pd.to_datetime(df['Date'])



# Introducing the Period column in format YYYY-MM

df['Month/Year'] = pd.to_datetime(df['Date']).dt.to_period('M')
df.info()
df.head()
print(df.groupby(['Region'])['AvgTemperature'].mean())

avg_temp_world = pd.Series(round(df.groupby('Region')['AvgTemperature'].mean().sort_values(),2))

avg_temp_world.plot(kind='bar', figsize = (10,6), color='yellow', alpha=0.5)

plt.xlabel('Mean Average Temperature')

plt.ylabel('Regions')

plt.title('Mean Average Temperatures by Region')
world_temp_date = pd.DataFrame(pd.Series(round(df.groupby('Date')['AvgTempCelcius'].mean(),2))[:-1])

world_temp_year = pd.DataFrame(pd.Series(round(df.groupby('Year')['AvgTempCelcius'].mean(),2))[:-1])



plt.subplot(2,1,1)

sns.set_style("darkgrid")

sns.lineplot(data = world_temp_date, color = 'blue')

plt.xlabel('Time')

plt.ylabel('Temperature (in Celcius)')

plt.title('Mean Avg. Temperature Over Time (Date) in the world')

plt.show()



plt.subplot(2,1,2)

sns.set_style("darkgrid")

sns.lineplot(data = world_temp_year, color = 'blue')

plt.xlabel('Time')

plt.ylabel('Temperature (in Celcius)')

plt.title('Mean Avg. Temperature Over Time (Years) in the world')

plt.show()
## Creating a function to plot the temperature over the Periods in different countries

def plot_temp_country_month(country, format = '-', temp = 'Celcius'):

    dat = df[df['Country'] == country]

    if temp == 'Celcius':

        dat_temp = pd.Series(round(dat.groupby('Date')['AvgTempCelcius'].mean().sort_values(),2))

    else:

        dat_temp = pd.Series(round(dat.groupby('Date')['AvgTemperature'].mean().sort_values(),2))

    sns.set_style("darkgrid")

    sns.lineplot(data = dat_temp, color = 'red')

    plt.xlabel('Time (Periods)')

    plt.ylabel('Temperature (in %s)' %temp)

    plt.title('Mean Avg. Temperature Over Time in %s' %country)

    plt.show()



    

## Creating a function to plot the temperature over the Years in different countries

def plot_temp_country_year(country, format = '-', temp = 'Celcius'):

    dat = df[df['Country'] == country]

    if temp == 'Celcius':

        dat_temp = pd.DataFrame(pd.Series(round(df[df['Country'] == country].groupby('Year')['AvgTempCelcius'].mean(),2))[:-1])

    else:

        dat_temp = pd.DataFrame(pd.Series(round(df[df['Country'] == country].groupby('Year')['AvgTemperature'].mean(),2))[:-1])

    sns.set_style("darkgrid")

    sns.lineplot(data = dat_temp, color = 'red' , style = 'event', hue = 'cue')

    plt.xlabel('Time (Years)')

    plt.ylabel('Temperature (in %s)' %temp)

    plt.title('Mean Avg. Temperature Over Time in %s' %country)

    plt.show()
plot_temp_country_year('India')

plot_temp_country_month('India')
## Function to calculate the temperature fluctuation

def calculate_fluctuation(series):

    fluctuation = np.zeros((len(series),))

    for i in range(1,len(series)):

        fluctuation[i] = series[i] - series[0]

    return fluctuation



## Function to plot the temperature fluctuation Lineplot

def plot_change(years, fluctuation, entity):

    change_df = pd.DataFrame(np.column_stack((years, fluctuation)), columns = ['Year', 'Change'])

    change_df['Year'] = change_df['Year'].astype(int)

    sns.lineplot(x = "Year", y = "Change", data = change_df, err_style="bars", ci=68, label = entity)

    x = np.zeros((len(change_df['Year']),1))

    plt.plot(change_df['Year'], x, '--')

    plt.title('Temperature fluctuations over the years')

    plt.ylabel('Change in Temperature')



## Function to plot the temperature fluctuation in the world 

def world_change_temp():

    dat = np.array(pd.Series(round(df.groupby('Year')['AvgTempCelcius'].mean(),2)))[:-1]

    years = np.arange(1995,2020)

    fluctuation = calculate_fluctuation(dat)

    plot_change(years, fluctuation, 'World')



## Function to plot the temperature fluctuation in every country

def country_change_temp(country):

    dat = np.array(pd.Series(round(df[df['Country'] == country].groupby('Year')['AvgTempCelcius'].mean(),2)))[:-1]

    years = np.arange(1995,2020)

    fluctuation = calculate_fluctuation(dat)

    plot_change(years, fluctuation, country)



## Function to plot the temperature fluctuation in every continent

def region_change_temp(region):

    dat = np.array(pd.Series(round(df[df['Region'] == region].groupby('Year')['AvgTempCelcius'].mean(),2)))[:-1]

    years = np.arange(1995,2020)

    fluctuation = calculate_fluctuation(dat)

    plot_change(years, fluctuation, region)



## Function to plot the average temperature and temperature fluctutaion distribution for every country

def temperature_histogram(country):

    hist1_s = np.array(pd.Series(round(df.groupby('Year')['AvgTempCelcius'].mean(),2)))[:-1]

    hist2_s = np.array(pd.Series(round(df[df['Country'] == country].groupby('Year')['AvgTempCelcius'].mean(),2)))[:-1]

    hist1_fluctuation = calculate_fluctuation(hist1_s)

    hist2_fluctuation = calculate_fluctuation(hist2_s)

    print('Skewness for Temperature in %s: ' %country, df[df['Country'] == country]['AvgTempCelcius'].skew())

    print('Kurtosis for Temperature in %s: ' %country, df[df['Country'] == country]['AvgTempCelcius'].kurt())

    plt.figure(figsize = (10,5))

    plt.subplot(1,2,1)

    sns.distplot(df[df['Country'] == country]['AvgTempCelcius'], label = country)

    sns.distplot(df['AvgTempCelcius'], label = 'World')

    plt.legend()

    plt.subplot(1,2,2)

    sns.distplot(hist1_fluctuation , label = 'World')

    sns.distplot(hist2_fluctuation, label = country)

    plt.xlabel('Temperature Fluctuations')

    plt.legend()
world_change_temp()
country_change_temp('US')
country_change_temp('Canada')
country_change_temp('China')
region_change_temp('Africa')
temperature_histogram('US')
# Create the series

series = np.array(list(pd.Series(round(df.groupby('Date')['AvgTempCelcius'].mean(),2))))



# Creating time intervals

time = np.array(np.arange(0,len(series)))
# We have 7000 training examples and rest as training examples

split_time = 7000



# Defining the Training set and test set

time_train = time[:split_time]

x_train = series[:split_time]

time_valid = time[split_time:]

x_valid = series[split_time:]



# Initialising the Hyperparamenters

window_size = 60

batch_size = 100

shuffle_buffer_size = 1000
## Function to create a line plot

def plot_series(time, series, format="-", start=0, end=None):

    plt.plot(time[start:end], series[start:end], format)

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.grid(False)



## Function to prepare data to be fed into the Tensorflow model

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

    series = tf.expand_dims(series, axis=-1)

    dp = tf.data.Dataset.from_tensor_slices(series)

    dp = dp.window(window_size + 1, shift=1, drop_remainder=True)

    dp = dp.flat_map(lambda w: w.batch(window_size + 1))

    dp = dp.shuffle(shuffle_buffer)

    dp = dp.map(lambda w: (w[:-1], w[1:]))

    return dp.batch(batch_size).prefetch(1)



## Function to prepare validation data into the model for prediction

def model_forecast(model, series, window_size):

    dp = tf.data.Dataset.from_tensor_slices(series)

    dp = dp.window(window_size, shift=1, drop_remainder=True)

    dp = dp.flat_map(lambda w: w.batch(window_size))

    dp = dp.batch(32).prefetch(1)

    forecast = model.predict(dp)

    return forecast
# Plotting the time series

plot_series(time, series)
## The model contains 1 ConV1D filter, 2 LSTMs, 3 Dense layers and 1 Lambda layer



tf.keras.backend.clear_session()

tf.random.set_seed(51)

np.random.seed(51)



## Defining window sizes for hyperparameter tuning 

window_size = 64

batch_size = 256

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

print(train_set)

print(x_train.shape)



# Defining the model

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv1D(filters=32, kernel_size=5,

                      strides=1, padding="causal",

                      activation="relu",

                      input_shape=[None, 1]),

  tf.keras.layers.LSTM(64, return_sequences=True),

  tf.keras.layers.LSTM(64, return_sequences=True),

  tf.keras.layers.Dense(30, activation="relu"),

  tf.keras.layers.Dense(10, activation="relu"),

  tf.keras.layers.Dense(1),

  tf.keras.layers.Lambda(lambda x: x * 400)

])



# Create a callback function to get optimal learning rate

lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-8 * 10**(epoch / 20))



# Defining the optimizer

optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)



# Compiling the model with Huber loss function 

model.compile(loss=tf.keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])



# Running the model

history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
plt.semilogx(history.history["lr"], history.history["loss"])

plt.axis([1e-8, 1e-3, 0, 60])
tf.keras.backend.clear_session()

tf.random.set_seed(51)

np.random.seed(51)

train_set = windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=shuffle_buffer_size)



# Defining the model with 1 ConV1D filter, 2 LSTMs, 3 Dense layers and 1 Lambda layer

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv1D(filters=60, kernel_size=5,

                      strides=1, padding="causal",

                      activation="relu",

                      input_shape=[None, 1]),

  tf.keras.layers.LSTM(60, return_sequences=True),

  tf.keras.layers.LSTM(60, return_sequences=True),

  tf.keras.layers.Dense(30, activation="relu"),

  tf.keras.layers.Dense(10, activation="relu"),

  tf.keras.layers.Dense(1),

  tf.keras.layers.Lambda(lambda x: x * 400)

])



# Defining the optimizer with learning rate 1e-6 and momentum 0.9

optimizer = tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)



# Compiling model with Huber loss function

model.compile(loss=tf.keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])



# Training with 50 epochs

history = model.fit(train_set,epochs=50)
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)

rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, rnn_forecast)
tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()