# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import datetime

import os

import seaborn as sns



import plotly.graph_objects as go

import plotly.express as pe

from fbprophet import Prophet

for dirname, _, filenames in os.walk('/kaggle/input/datathon'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Display the 3 dataframes

production = pd.read_csv("/kaggle/input/datathon/Total_Electricity_Production.csv")

print(production.shape)

display(production)



indicators = pd.read_csv("/kaggle/input/datathon/world_indicators.csv")

print(indicators.shape)



display(indicators)



history = pd.read_csv("/kaggle/input/datathon/south_africa_load_shedding_history.csv")

print(history.shape)



display(history)

production = pd.read_csv("/kaggle/input/datathon/Total_Electricity_Production.csv")



production.columns = ["datetime","production"]

production["datetime"] = pd.to_datetime(production["datetime"])

production["year"] = production["datetime"].apply(lambda x: x.year)

production["month"] = production["datetime"].apply(lambda x: x.month)

production["date"] = production["datetime"].apply(lambda x: x.date())

production["day_of_week"] = production["datetime"].apply(lambda x: x.day)

production["week"] = production["datetime"].apply(lambda x: x.week)

import plotly.graph_objects as go

fig=go.Figure()





fig.add_trace(go.Scatter(x=production["datetime"], y=production["production"],

                    mode='lines+markers',

                    name='Production'))

fig.update_layout(title="Total Electricity Production in South Africa, exisiting data",

                  xaxis_title="Datetime",yaxis_title="Giggawatts Hours")



fig.show()

elec = production.copy()

display(production)
fig=go.Figure()





fig.add_trace(go.Scatter(x=production.loc[production["month"] == 7]["datetime"], y=production.loc[production["month"] == 7]["production"],

                    mode='lines+markers',

                    name='Production'))

fig.update_layout(title="Total Electricity Production in South Africa July of Each Year, exisiting data",

                  xaxis_title="Datetime",yaxis_title="Giggawatts Hours")



fig.show()

elec = production.copy()
production_by_day = pd.DataFrame(index = pd.date_range(start='1985-01-01',end="2018-07-01",freq='D'))

production_by_day["production"] = production_by_day.index

elec

production_dict = production[["datetime","production"]].set_index("datetime")["production"].to_dict()

production_by_day["production"] = production_by_day["production"].replace(production_dict)

production_by_day["production"] = production_by_day["production"].apply(lambda x:np.nan if (type(x) != float) else x)

#df["production"] = df['production'].interpolate(method='time', inplace=True)

#df["production"] 

production_by_day["production"].interpolate(method='time', inplace=True)

production_by_day["production"]

import plotly.graph_objects as go

fig=go.Figure()





fig.add_trace(go.Scatter(x=production_by_day.index, y=production_by_day["production"],

                    mode='lines+markers',

                    name='Production'))

fig.update_layout(title="Total Electricity Production in South Africa,estimated using imputation",

                  xaxis_title="Datetime",yaxis_title="Giggawatts Hours")



fig.show()
X = production_by_day.reset_index()

X.columns = ["ds","y"]

m = Prophet(weekly_seasonality=False)

m.add_seasonality(name='yearly', period=365, fourier_order=10)

test = pd.DataFrame({"ds":pd.date_range(start='2018-07-02',end="2021-09-07",freq='D')})



elec_forecast = m.fit(X).predict(test)

#fig = m.plot_components(forecast)



#from fbprophet.plot import plot_plotly, plot_components_plotly

m.plot(elec_forecast)
indicators = pd.read_csv("../input/datathon/world_indicators.csv")

indicators.columns = indicators.columns.str.lower()

indicators = indicators.applymap(lambda s:s.lower() if type(s) == str else s)

display(indicators.loc[225:232])

indicators = indicators.drop(range(228,233),axis=0)

indicators = indicators.drop(["series code","country code"],axis=1)

indicators.columns = indicators.columns.str.replace("\s\[.*\]","")



tuples = zip(indicators["country name"],indicators["series name"])

indicators.index = pd.MultiIndex.from_tuples(tuples, names=['country', 'indicator'])

indicators=indicators.drop(["country name","series name"],axis=1)

indicators
# A function that takes information of a particular country of particular series

def get_country_df(country_name,df,col):

    

    

    sa_indicators = df.loc[country_name]

    sa_indicators=sa_indicators.transpose()

    return sa_indicators[col].dropna()

ser = get_country_df("south africa",indicators,"access to electricity (% of population)")

indicators.index.get_level_values(1).unique()
country_names = np.unique(indicators.index.get_level_values(0))

fig = go.Figure()

graph_names=["access to electricity, rural (% of rural population)",

            "access to electricity, urban (% of urban population)",

            "access to electricity (% of population)"]

for name in graph_names:

    ser = get_country_df("south africa",indicators,name)

    fig.add_trace(go.Scatter(y=ser.values, x=ser.index,

                    mode='lines+markers',

                    name=name))

fig.update_layout(title = "access to electricity of South Africa",xaxis_title = "year",yaxis_title="percentage")



fig.show()

# comparison of Overall Access



fig = go.Figure()



for country in country_names:

    ser_access = get_country_df(country,indicators,"access to electricity (% of population)")

    #ser_rural_access = get_country_df(country,indicators,"access to electricity, rural (% of rural population)")

    #ser_urban_access = get_country_df(country,indicators,"access to electricity, urban (% of urban population)")

    #ser_gdp = get_country_df(country,indicators,"gdp per capita (constant 2010 us$)")

    fig.add_trace(go.Scatter(y=ser_access.values, x=ser_access.index,

                    mode='lines+markers',

                    name=country))

fig.update_layout(title = "Access to Electricity (% of Population)",xaxis_title = "Year",yaxis_title="Percentage (%)")

fig.show()



# comparison of Urban Access



fig = go.Figure()



for country in country_names:

    ser = get_country_df(country,indicators,"access to electricity, urban (% of urban population)")

    fig.add_trace(go.Scatter(y=ser.values, x=ser.index,

                    mode='lines+markers',name=country))

fig.update_layout(title = "Access to Electricity, Urban (% of Urban Population)",xaxis_title = "Year",yaxis_title="Percentage (%)")

fig.show()



# comparison of Urban Access



fig = go.Figure()



for country in country_names:

    ser = get_country_df(country,indicators,"access to electricity, rural (% of rural population)")

    fig.add_trace(go.Scatter(y=ser.values, x=ser.index,

                    mode='lines+markers',name=country))

fig.update_layout(title = "Access to Electricity, Rural (% of Rural Population)",xaxis_title = "Year",yaxis_title="Percentage (%)")

fig.show()



fig = go.Figure()



for country in country_names:

    ser = get_country_df(country,indicators,"time required to get electricity (days)")

    fig.add_trace(go.Scatter(y=ser.values, x=ser.index,

                    mode='lines+markers',

                    name=country))

fig.update_layout(title = "time required to get electricity (days) in South Africa and its neighbors",xaxis_title = "year",yaxis_title="days")



fig.show()



fig = go.Figure()



for country in country_names:

    ser = get_country_df(country,indicators,"electricity production from oil, gas and coal sources (% of total)")

    fig.add_trace(go.Scatter(y=ser.values, x=ser.index,

                    mode='lines+markers',

                    name=country))

fig.update_layout(title = "Electricity Production From oil, Gas and Coal Sources (% of total)",xaxis_title = "year",yaxis_title="Percentage (%)")



fig.show()

fig = go.Figure()



for country in country_names:

    ser = get_country_df(country,indicators,"electricity production from renewable sources, excluding hydroelectric (% of total)")

    fig.add_trace(go.Scatter(y=ser.values, x=ser.index,

                    mode='lines+markers',

                    name=country))

fig.update_layout(title = "Electricity Production from Renewable Sources, Excluding Hydroelectric (% of total)",xaxis_title = "year",yaxis_title="Percentage (%)")



fig.show()

ser1= get_country_df("world",indicators,"electricity production from renewable sources, excluding hydroelectric (% of total)").loc["2015"]

ser2= get_country_df("world",indicators,"electricity production from oil, gas and coal sources (% of total)").loc['2015']

# This dataframe has 244 lines, but 4 distinct values for `day`

pie = pd.DataFrame([float(ser1),float(ser2)])

pie.index = ["electricity production from renewable sources, excluding hydroelectric (% of total)","electricity production from oil, gas and coal sources (% of total)"]

pie.columns= ["percentage"]

fig = pe.pie(pie.reset_index(), values='percentage', names='index')

fig.update_layout(title="Eletricity Production Composition in 2015, World Average")

fig.show()



ser1= get_country_df("south africa",indicators,"electricity production from renewable sources, excluding hydroelectric (% of total)").loc["2015"]

ser2= get_country_df("south africa",indicators,"electricity production from oil, gas and coal sources (% of total)").loc['2015']

# This dataframe has 244 lines, but 4 distinct values for `day`

pie = pd.DataFrame([float(ser1),float(ser2)])

pie.index = ["electricity production from renewable sources, excluding hydroelectric (% of total)","electricity production from oil, gas and coal sources (% of total)"]

pie.columns= ["percentage"]

fig = pe.pie(pie.reset_index(), values='percentage', names='index')

fig.update_layout(title="Eletricity Production Composition in 2015, South Africa")

fig.show()

history = pd.read_csv("../input/datathon/south_africa_load_shedding_history.csv")

history.columns = ["datetime","stage"]

history["datetime"] = pd.to_datetime(history["datetime"])

history["year"] = history["datetime"].apply(lambda x: x.year)

history["month"] = history["datetime"].apply(lambda x: x.month)

history["date"] = history["datetime"].apply(lambda x: x.date()).astype(str)

history["day_of_week"] = history["datetime"].apply(lambda x: x.day)

history["week"] = history["datetime"].apply(lambda x: x.week)



history_by_day = pd.DataFrame(index = pd.date_range(start='2015-01-09',end="2020-09-07",freq='D'))

history_dic = history.groupby("date").agg({"stage":"sum"})["stage"].to_dict()

history_by_day["1000_MW"] = history_by_day.index.astype(str)

history_by_day["1000_MW"] = history_by_day["1000_MW"].replace(history_dic)

history_by_day["1000_MW"] = history_by_day["1000_MW"].apply(lambda x:0 if (type(x) != int) else x)

display(history_by_day)

fig=go.Figure()

fig.update_layout(title="Energy to Be Shed Each Day from 2015 to 2020",

                 xaxis_title="Date",yaxis_title="Energy (MW)",legend=dict(x=0,y=1,traceorder="normal"))

fig.add_trace(go.Scatter(x=history_by_day.index,

                         y=history_by_day["1000_MW"] * 1000,

                    mode='lines+markers',

                    name='Outage'))





fig.show()

# To see a smoother curve, we use 7-day rolling or 30-day rolling. 

fig=go.Figure()



fig.add_trace(go.Scatter(x=history_by_day.index, y=(history_by_day["1000_MW"].rolling(window=7).mean())*1000,

                    mode='lines+markers',

                    name='Weekly'))

fig.update_layout(title="7-Day Rolling Mean",

                 xaxis_title="Date",yaxis_title="Energy (MW)",legend=dict(x=0,y=1,traceorder="normal"))



fig.show()

fig=go.Figure()



fig.add_trace(go.Scatter(x=history_by_day.index, y=1000*history_by_day["1000_MW"].rolling(window=30).mean(),

                    mode='lines+markers',

                    name='Monthly'))



fig.update_layout(title="30-day Rolling Mean",

                 xaxis_title="Date",yaxis_title="Energy (MW)",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
history_count = pd.DataFrame(index = pd.date_range(start='2015-01-01',end="2020-09-07",freq='D'))

history_count["num_blackouts"] = history_count.index.astype(str)

dic_count= history.groupby("date").agg({"stage":"count"})["stage"].to_dict()

history_count["num_blackouts"] = history_count["num_blackouts"].replace(dic_count)



history_count["num_blackouts"] = history_count["num_blackouts"].apply(lambda x:0 if (type(x) != int) else x)

year_sum = history.groupby("year").agg({"stage":"count"})

year_sum.columns = ["num_blackouts_sum"]





year_sum = year_sum.reset_index()

#outage.groupby("year").agg({"stage":"count"})

year_sum.columns = ["year","Number of national blackouts"]

ax = sns.barplot(x="year", y="Number of national blackouts", data=year_sum)





year_sum = history.groupby("year").agg({"stage":"sum"})

year_sum.columns = ["energy_load_in_MW"]

year_sum["energy_load_in_MW"] = year_sum["energy_load_in_MW"] * 1000



year_sum = year_sum.reset_index()

year_sum.columns = ["year","energy_load_in_MW"]

ax = sns.barplot(x="year", y="energy_load_in_MW", data=year_sum)
# Weekly Sum

history_week = history_by_day.reset_index().resample('W-Wed', label='right', closed = 'right', on='index').sum().reset_index()

fig=go.Figure()



fig.add_trace(go.Scatter(x=history_week["index"], y=history_week["1000_MW"] * 1000,

                    mode='lines+markers',

                    name='Weekly'))



fig.update_layout(title="National Energy Load to be Shed By Week",

                 xaxis_title="Date",yaxis_title="Energy (MW)",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
elec_forecast
elec_features = elec_forecast[["ds","trend","yhat"]]

elec_features.set_index("ds")

test = pd.DataFrame({"ds":pd.date_range(start='2018-06-01',end="2021-09-07",freq='D')})



elec_forecast_df = m.predict(test)

elec_testing_features = elec_forecast_df[["ds","trend","yhat"]]

elec_testing_features.set_index("ds")







elec_training_features = elec_features[["ds","trend","yhat"]]

display(elec_training_features)

#elec_training_features.set_index("ds")
df_training = pd.DataFrame(index = pd.date_range(start='2018-06-01',end="2020-09-07",freq='D'))



df_training["1000_MW"] = df_training.index.astype(str)

df_training["date"] = df_training.index

dic = history.groupby("date").agg({"stage":"sum"})["stage"].to_dict()

df_training["1000_MW"] = df_training["1000_MW"].replace(dic)

df_training["1000_MW"] = df_training["1000_MW"].apply(lambda x:0 if (type(x) != int) else x)

elec_training_features

training = pd.concat([df_training.reset_index(),elec_training_features],axis=1).drop(["index","ds"],axis=1)

#training = training.resample('W-Wed', label='right', closed = 'right', on="date").sum().reset_index()

training = training.drop("date",axis = 1).dropna()

training

#training.to_csv("training.csv",index=False)


testing = elec_testing_features.copy()

testing["ds"] = 0

testing.columns = ["1000_MW","trend","yhat"]



combined = pd.concat([training,testing[830:1195]],ignore_index=True)

display(combined)
from hmmlearn import hmm

import matplotlib.pyplot as plt

import warnings

regr = hmm.GaussianHMM(n_components = 7,n_iter = 1000)

with warnings.catch_warnings():

    warnings.simplefilter('ignore')

    regr.fit(training)

#plt.plot(regr.predict(training))

plt.plot(training["1000_MW"])

#combined = combined.drop("ds",axis = 1)

plt.plot(regr.predict(combined))



predictions = regr.predict(combined)

future_predictions = predictions[830:]

future_dates = elec_testing_features.loc[830:][["ds"]]

future_dates["preds"] = future_predictions

fig = go.Figure()

fig.add_trace(go.Scatter(x=future_dates.ds, y=future_dates.preds * 1000,

                    mode='lines+markers',

                    name='Production'))

fig.update_layout(title="2020-2021 Predicted Energy Being Shedded using HMM",

                  xaxis_title="Datetime",yaxis_title="Energy (MW)")



fig.show()
# Sum of energy for next year, predicted

print("Predicted Amount of Energy Load to be shed for 2020-2021:")

print(str(np.sum(predictions[830:]) * 1000)+" MW")

past_3_years = history.loc[history["year"]>2015]

df_peak= pd.DataFrame(index = pd.date_range(start='2017-09-07',end="2020-09-07",freq='D'))



df_peak["num_blackouts"] = df_peak.index.astype(str)

dic = history.groupby("date").agg({"stage":"count"})["stage"].to_dict()

df_peak["num_blackouts"] = df_peak["num_blackouts"].replace(dic)

df_peak["num_blackouts"] = df_peak["num_blackouts"].apply(lambda x:0 if (type(x) != int) else x)

df_peak
import matplotlib.pyplot as plt

from scipy.misc import electrocardiogram

from scipy.signal import find_peaks,peak_widths

df_peak = df_peak.reset_index()

x = df_peak["num_blackouts"]

peaks, _ = find_peaks(x, height=0)

plt.plot(x)

plt.plot(peaks, x[peaks], "x")

plt.plot(np.zeros_like(x), "--", color="gray")

plt.show()
from scipy.signal import chirp, find_peaks, peak_widths

import matplotlib.pyplot as plt

x = df_peak["num_blackouts"]



peaks, _ = find_peaks(x)

results_half = peak_widths(x, peaks, rel_height=0.5)



results_full = peak_widths(x, peaks, rel_height=1)



plt.plot(x)

plt.plot(peaks, x[peaks], "x")

plt.hlines(*results_half[1:], color="C2")

plt.hlines(*results_full[1:], color="C3")

plt.show()
avg_blackouts_per_time = np.mean(df_peak.loc[df_peak["num_blackouts"]!=0]["num_blackouts"])

avg_blackout_length = np.mean(results_full[0])

past_peak_num = len(peaks)

avg_peak = np.mean(x[peaks].values)

#print(df_peak["num_blackouts"].sum())

print("Prediction for the total number of blackouts for next year:"+ str(past_peak_num*avg_blackout_length*avg_blackouts_per_time/3))