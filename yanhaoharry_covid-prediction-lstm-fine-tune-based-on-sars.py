import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import requests

import warnings



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib.pyplot as plt

import matplotlib

import matplotlib.dates as mdates

import seaborn as sns

import plotly.express as px

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras import layers

from keras.models import model_from_json



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_covid19=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df_covid19.drop(['SNo','Last Update','Province/State'],axis=1,inplace = True)

df_covid19['ObservationDate']=pd.to_datetime(df_covid19['ObservationDate'])
# Because there are several updates in the same day, we need a groupby function to merge the data in the same day 

df_covid19 = df_covid19.groupby(["ObservationDate","Country/Region"],as_index = False).sum()

df_covid19_compare = df_covid19

df_covid19 = df_covid19.set_index('ObservationDate')

df_covid19.tail()
df_sars = pd.read_csv('../input/sars-2003-complete-dataset-clean/sars_2003_complete_dataset_clean.csv')

df_sars.rename(columns={'Date':'ObservationDate', 'Country':'Country/Region', 'Cumulative number of case(s)':'Confirmed', 'Number of deaths':'Deaths','Number recovered':'Recovered' }, inplace=True)

df_sars['ObservationDate']=pd.to_datetime(df_sars['ObservationDate'])

df_sars_compare = df_sars

df_sars = df_sars.set_index('ObservationDate')
Sars_CA = df_sars[df_sars['Country/Region'] == 'China']

Sars_CA.tail()
covid19_new=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covid19_new['Active'] = covid19_new['Confirmed'] - covid19_new['Deaths'] - covid19_new['Recovered']

covid19_new["ObservationDate"] = pd.to_datetime(covid19_new["ObservationDate"])

print("Active Cases Column Added Successfully")

covid19_new.head()
wep = covid19_new.groupby(["ObservationDate","Country/Region"])["Confirmed","Deaths","Recovered"].max()

wep = wep.reset_index()

wep["ObservationDate"] = wep["ObservationDate"].dt.strftime("%m,%d,%Y")

wep["Country"] = wep["Country/Region"]



choro_map = px.choropleth(wep, 

                          locations= "Country", 

                          locationmode = "country names",

                          color = "Confirmed", 

                          hover_name = "Country/Region",

                          projection = "natural earth",

                          animation_frame = "ObservationDate",

                          color_continuous_scale = "Blues",

                          range_color = [10000,200000])

choro_map.update_layout(

    title_text = 'Global Spread of Coronavirus',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

choro_map.show()
covid19_new.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country', 'Province/State':'Province' }, inplace=True)

covid19_new['Date']=pd.to_datetime(covid19_new['Date'])



maxdate=max(covid19_new['Date'])



fondate=maxdate.strftime("%Y-%m-%d")

print("The last observation date is {}".format(fondate))

ondate = format(fondate)
date_list1 = list(covid19_new["Date"].unique())

confirmed = []

deaths = []

recovered = []

active = []

for i in date_list1:

    x = covid19_new[covid19_new["Date"] == i]

    confirmed.append(sum(x["Confirmed"]))

    deaths.append(sum(x["Deaths"]))

    recovered.append(sum(x["Recovered"]))

    active.append(sum(x["Active"]))

data_glob = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])

data_glob.tail()
import plotly.graph_objs as go 

trace1 = go.Scatter(

x = data_glob["Date"],

y = data_glob["Confirmed"],

mode = "lines",

name = "Confirmed",

line = dict(width = 2.5),

marker = dict(color = [0, 1, 2, 3])

)



trace2 = go.Scatter(

x = data_glob["Date"],

y = data_glob["Deaths"],

mode = "lines",

name = "Deaths",

line = dict(width = 2.5),

marker = dict(color = [0, 1, 2, 3])

)



trace3 = go.Scatter(

x = data_glob["Date"],

y = data_glob["Recovered"],

mode = "lines",

name = "Recovered",

line = dict(width = 2.5),    

marker = dict(color = [0, 1, 2, 3])

)



trace4 = go.Scatter(

x = data_glob["Date"],

y = data_glob["Active"],

mode = "lines",

name = "Active",

line = dict(width = 2.5),

marker = dict(color = [0, 1, 2, 3])

)



data_plt = [trace1,trace2,trace3,trace4]

layout = go.Layout(title = "Global Case States",xaxis_title="Date",yaxis_title="Number of Total Cases",

                   legend=dict(

        x=0,

        y=1,),hovermode='x')

fig = go.Figure(data = data_plt,layout = layout)



fig.show()
labels = ["Recovered","Deaths","Active"]

values = [data_glob.tail(1)["Recovered"].iloc[0],data_glob.tail(1)["Deaths"].iloc[0],data_glob.tail(1)["Active"].iloc[0]]



fig = go.Figure(data = [go.Pie(labels = labels, values = values,textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "Global Patient Percentage"))

fig.show()
df_covid19_compare.info()
df_sars_compare.info()
date_list_cov_compare = list(df_covid19_compare["ObservationDate"].unique())

confirmed = []

deaths = []

recovered = []

for i in date_list_cov_compare:

    x = df_covid19_compare[df_covid19_compare["ObservationDate"] == i]

    confirmed.append(sum(x["Confirmed"]))

    deaths.append(sum(x["Deaths"]))

    recovered.append(sum(x["Recovered"]))

data_glob_cov = pd.DataFrame(list(zip(date_list_cov_compare,confirmed,deaths,recovered)),columns = ["Date","Confirmed","Deaths","Recovered"])

data_glob_cov.tail()
date_list_sars_compare = list(df_sars_compare["ObservationDate"].unique())

confirmed = []

deaths = []

recovered = []

for i in date_list_sars_compare:

    x = df_sars_compare[df_sars_compare["ObservationDate"] == i]

    confirmed.append(sum(x["Confirmed"]))

    deaths.append(sum(x["Deaths"]))

    recovered.append(sum(x["Recovered"]))

data_glob_sars = pd.DataFrame(list(zip(date_list_sars_compare,confirmed,deaths,recovered)),columns = ["Date","Confirmed","Deaths","Recovered"])

data_glob_sars.tail()
from plotly import subplots

death_percent_sars = ((data_glob_sars["Deaths"]*100)/data_glob_sars["Confirmed"])

death_percent_cov = ((data_glob_cov["Deaths"]*100)/data_glob_cov["Confirmed"])



trace_death_sars = go.Scatter(x=data_glob_sars["Date"],

                                  y = death_percent_sars,

                                  mode = "lines",

                                  name = "Death Percentage for SARS",

                                  marker = dict(color = [0, 1, 2, 3]))

    

trace_death_cov = go.Scatter(x=data_glob_cov["Date"],

                                  y = death_percent_cov,

                                  mode = "lines",

                                  name = "Death Percentage for Covid-19",

                                  marker = dict(color = [0, 1, 2, 3]))

    

death_plt = [trace_death_sars,trace_death_cov]



fig = subplots.make_subplots(rows=1,cols=2)

fig.append_trace(trace_death_sars,1,1)

fig.append_trace(trace_death_cov,1,2)



fig.layout.width = 1000

fig.layout.height = 600

fig.show()
recover_percent_sars = ((data_glob_sars["Recovered"]*100)/data_glob_sars["Confirmed"])

recover_percent_cov = ((data_glob_cov["Recovered"]*100)/data_glob_cov["Confirmed"])



trace_recover_sars = go.Scatter(x=data_glob_sars["Date"],

                                  y = recover_percent_sars,

                                  mode = "lines",

                                  name = "Recover Percentage for SARS",

                                  marker = dict(color = [0, 1, 2, 3]))

    

trace_recover_cov = go.Scatter(x=data_glob_cov["Date"],

                                  y = recover_percent_cov,

                                  mode = "lines",

                                  name = "Recover Percentage for Covid-19",

                                  marker = dict(color = [0, 1, 2, 3]))

    

recover_plt = [trace_recover_sars,trace_recover_cov]



fig = subplots.make_subplots(rows=1,cols=2)

fig.append_trace(trace_recover_sars,1,1)

fig.append_trace(trace_recover_cov,1,2)

fig.layout.width = 1000

fig.layout.height = 600

fig.show()
# load Sars data set, and set the country as China

Sars_CA.tail()
# data normalization



train_num_sars = int(len(Sars_CA)*0.8)



scaler_sars = MinMaxScaler()



train_origin = pd.DataFrame(Sars_CA.iloc[:train_num_sars,1])

test_origin = pd.DataFrame(Sars_CA.iloc[train_num_sars:,1])



scaler_sars.fit(train_origin)

scaled_train_sars = scaler_sars.transform(train_origin)

scaled_test_sars = scaler_sars.transform(test_origin)

# using 10 day lag to predict the model

n_input = 15

n_features = 1

generator_sars = TimeseriesGenerator(scaled_train_sars, scaled_train_sars, length=n_input, batch_size=1)
# show the format of our input data

for i in range(3):

    x, y = generator_sars[i]

    print('%s => %s' % (x, y))
# build 4-layer RNN model

# define model

model = Sequential([

    layers.LSTM(256, activation='relu', input_shape=(n_input, n_features),return_sequences=True),

    layers.LSTM(128, activation='relu', input_shape=(n_input, n_features),return_sequences=True),

    layers.LSTM(64, activation='relu', input_shape=(n_input, n_features)),

    layers.Dense(1)

])



model.compile(optimizer='adam', loss='mse')

model.summary()
# train the model

model.fit_generator(generator_sars,epochs=25)
# plot the loss curve

loss_per_epoch = model.history.history['loss']

fig = plt.figure(dpi = 120,figsize = (6,4))

ax = plt.axes()

ax.set(xlabel = 'Number of Epochs',ylabel = 'MSE Loss',title = 'Sars Loss Curve')

plt.plot(range(len(loss_per_epoch)),loss_per_epoch,lw = 1);
# test our model in test set

test_predictions = []



first_eval_batch = scaled_train_sars[-n_input:]

current_batch = first_eval_batch.reshape((1, n_input, n_features))



for i in range(len(test_origin)):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model.predict(current_batch)[0]

    

    # store prediction

    test_predictions.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
# fill test table with prediction

true_predictions = scaler_sars.inverse_transform(test_predictions)

test_origin['Predictions'] = true_predictions

print(test_origin)
# plot the comparison between actual value and predicted value

fig = plt.figure(dpi = 120)

ax=plt.axes()

test_origin.plot(legend=True,figsize=(6,4),lw = 2,ax=ax)

plt.xlabel('Date')

plt.ylabel('Count of Cases')

plt.title('Comparision Test and Prediction')

plt.show();
# build complete SARS model with the whole data set (train+test)

scaler_sars = MinMaxScaler()



train_origin = pd.DataFrame(Sars_CA.iloc[:,1])





scaler_sars.fit(train_origin)

scaled_train_sars = scaler_sars.transform(train_origin)



n_input = 15

n_features = 1

generator_sars = TimeseriesGenerator(scaled_train_sars, scaled_train_sars, length=n_input, batch_size=1)



# define model

model_whole = Sequential([

    layers.LSTM(256, activation='relu', input_shape=(n_input, n_features),return_sequences=True),

    layers.LSTM(128, activation='relu', input_shape=(n_input, n_features),return_sequences=True),

    layers.LSTM(64, activation='relu', input_shape=(n_input, n_features)),

    layers.Dense(1)

])

model_whole.compile(optimizer='adam', loss='mse')



# fit model

model_whole.fit_generator(generator_sars,epochs=25)
# plot loss curve for complete model

loss_per_epoch = model_whole.history.history['loss']

fig = plt.figure(dpi = 120,figsize = (6,4))

ax = plt.axes()

ax.set(xlabel = 'Number of Epochs',ylabel = 'MSE Loss',title = 'Loss Curve of Base Model')

plt.plot(range(len(loss_per_epoch)),loss_per_epoch,lw = 2)

fig.show()
# save SARS model and its weights

json_config = model_whole.to_json()

with open('model_config.json', 'w') as json_file:

    json_file.write(json_config)

model_whole.save_weights('path_to_my_weights.h5')
# load COVID-19 dataset, and set country as Canada

Covid_CA = df_covid19[df_covid19['Country/Region'] == 'Canada']

Covid_CA.head()
# normalization



train_num = int(len(Covid_CA)*0.8)



scaler = MinMaxScaler()



train = pd.DataFrame(Covid_CA.iloc[:train_num,1])

test = pd.DataFrame(Covid_CA.iloc[train_num:,1])



scaler.fit(train)

scaled_train = scaler.transform(train)

scaled_test = scaler.transform(test)

# show confirmation cases after normalization 

print("Scaled Train Set:", scaled_train[:3],"\n")

print("Scaled Test Set:", scaled_test[:3])
# equally, we set 10 day lag for modelling

n_input = 15

n_features = 1

generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
# show data format

for i in range(3):

    x, y = generator[i]

    print('%s => %s' % (x, y))
# load SARS model

model_cov = model_from_json(open('model_config.json').read())

model_cov.load_weights('path_to_my_weights.h5')

model_cov.summary()
# Let's take a look to see how many layers are in the base model

print("Number of layers in the base model: ", len(model_cov.layers))



# Fine-tune from this layer onwards

fine_tune_at = 1



# Freeze all the layers before the `fine_tune_at` layer

for layer in model_cov.layers[:fine_tune_at]:

  layer.trainable =  False
# now, we locked the first one layers of our network. The Non-trainable params is 66560+49408

model_cov.summary()
# compile the new model for training

model_cov.compile(optimizer='adam', loss='mse')
# fit model

model_cov.fit_generator(generator,epochs=25)
# plot loss curve for new model

loss_per_epoch = model_cov.history.history['loss']

fig = plt.figure(dpi = 120,figsize = (6,4))

ax = plt.axes()

ax.set(xlabel = 'Number of Epochs',ylabel = 'MSE Loss',title = 'Loss Curve - Fine Tuning')

plt.plot(range(len(loss_per_epoch)),loss_per_epoch,lw = 2);
test_predictions = []



first_eval_batch = scaled_train[-n_input:]

current_batch = first_eval_batch.reshape((1, n_input, n_features))



for i in range(len(test)):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model_cov.predict(current_batch)[0]

    

    # store prediction

    test_predictions.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = scaler.inverse_transform(test_predictions)

test['Predictions'] = true_predictions

test.head()
fig = plt.figure(dpi = 120)

ax=plt.axes()

test.plot(legend=True,figsize=(6,4),lw = 2,ax=ax)

plt.xlabel('Date')

plt.ylabel('Count of Cases')

plt.show();