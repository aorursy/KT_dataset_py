import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import plotly.offline as py

import statsmodels

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import pandas_profiling

from math import sqrt

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score, TimeSeriesSplit

import statsmodels.api as sm

import matplotlib.animation as animation

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/earthquakes-in-turkey/Catalogue.csv",encoding = "ISO-8859-1")

df=df.rename(columns={"Enlem": "latitude", "Boylam": "longtitude", "Büyüklük":"Magnitude", "Derinlik":"Depth", "Zaman (UTC)":"Time", "Tip":"Type"})

df['Time']=pd.to_datetime(df['Time'], errors='coerce')

df.columns
df=df[['Time',  'latitude', 'longtitude', 'Magnitude', 'Depth', 'Type' ]]
df.profile_report()
fig = go.Figure(go.Densitymapbox(lat=df.latitude, lon=df.longtitude, z=df.Magnitude, radius=5))

fig.update_layout(mapbox_style="stamen-terrain")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.line(df, x='Time', y='Magnitude')

fig.show()
#And depth:

fig = px.line(df, x='Time', y='Depth')

fig.show()
fig=plt.figure()

ax=fig.add_subplot(111)

ax.plot(df['Magnitude'], df['Depth'], '*')

plt.axvspan(6, 8, color='red', alpha=0.5)

plt.xlabel('magnitude')

plt.ylabel('depth')

plt.show()
data=pd.DataFrame(df['Magnitude'].values, index=df['Time'])

data.columns = ["y"]
for i in range(150, 500): 

    data["lag_{}".format(i)] = data.y.shift(i)    
def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#--->>https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python

def timeseries_train_test_split(X, y, test_size):

    test_index = int(len(X)*(1-test_size))

    """

        Perform train-test split with respect to time series structure

    """

    X_train = X.iloc[:test_index]

    y_train = y.iloc[:test_index]

    X_test = X.iloc[test_index:]

    y_test = y.iloc[test_index:]

    

    return X_train, X_test, y_train, y_test
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

y = data.y

X = data.drop(['y'], axis=1)



X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
#--->>https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python

tscv = TimeSeriesSplit(n_splits=3) 

def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):

    prediction = model.predict(X_test)

    

    plt.figure(figsize=(15, 7))

    plt.plot(prediction, "g", label="prediction", linewidth=5.0)

    plt.plot(y_test.values, label="actual", linewidth=2.0)

    

    if plot_intervals:

        cv = cross_val_score(model, X_train, y_train, 

                                    cv=tscv, 

                                    scoring="neg_mean_absolute_error")

        mae = cv.mean() * (-1)

        deviation = cv.std()

        

        scale = 1.96

        lower = prediction - (mae + scale * deviation)

        upper = prediction + (mae + scale * deviation)

        

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)

        plt.plot(upper, "r--", alpha=0.5)

        

        if plot_anomalies:

            anomalies = np.array([np.NaN]*len(y_test))

            anomalies[y_test<lower] = y_test[y_test<lower]

            anomalies[y_test>upper] = y_test[y_test>upper]

            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

    

    error = mean_absolute_percentage_error(prediction, y_test)

    plt.title("Mean absolute percentage error {0:.2f}%".format(error))

    plt.legend(loc="best")

    plt.tight_layout()

    plt.grid(True);
from xgboost import XGBRegressor 

xgb = XGBRegressor()

xgb.fit(X_train_scaled, y_train)
plotModelResults(xgb, 

                 X_train=X_train_scaled, 

                 X_test=X_test_scaled, 

                 plot_intervals=True, plot_anomalies=False)