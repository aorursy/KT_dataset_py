# Basic libraries

import numpy as np

import pandas as pd

import warnings

warnings.simplefilter('ignore')



# Directry check

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# file

import zipfile



# Data preprocessing

import datetime

import re

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# Visualization

from matplotlib import pyplot as plt

import folium

import seaborn as sns



# Random forest

from sklearn.ensemble import RandomForestRegressor

from sklearn.multioutput import MultiOutputRegressor



# parameter opimization

from sklearn.model_selection import GridSearchCV



# Validation

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
## Dataloading
# sample

zip_file = zipfile.ZipFile("/kaggle/input/pkdd-15-predict-taxi-service-trajectory-i/sampleSubmission.csv.zip")

sample = pd.read_csv(zip_file.open('sampleSubmission.csv'))
# train

zip_file = zipfile.ZipFile("/kaggle/input/pkdd-15-predict-taxi-service-trajectory-i/train.csv.zip")

train_df = pd.read_csv(zip_file.open("train.csv"))
# test

zip_file = zipfile.ZipFile("/kaggle/input/pkdd-15-predict-taxi-service-trajectory-i/test.csv.zip")

test_df = pd.read_csv(zip_file.open("test.csv"))
# location_Data

zip_file = zipfile.ZipFile("/kaggle/input/pkdd-15-predict-taxi-service-trajectory-i/metaData_taxistandsID_name_GPSlocation.csv.zip")

loc_df = pd.read_csv(zip_file.open("metaData_taxistandsID_name_GPSlocation.csv"))
# sample_submission

sample.head()
# test_data

test_df.head()
# test_data size

test_df.shape
# train_data

train_df.head()
# train_data size

train_df.shape
# train_data info

train_df.info()
# train_data null data

train_df.isnull().sum()
# train_data unique values

for i in range(train_df.shape[1]):

    print('*'*50)

    print(train_df.columns[i])

    print(train_df.iloc[:,i].value_counts())
# test_data unique values

for i in range(test_df.shape[1]):

    print('*'*50)

    print(test_df.columns[i])

    print(test_df.iloc[:,i].value_counts())
# Time data preprocessing

train_df["TIMESTAMP"] = [float(time) for time in train_df["TIMESTAMP"]]

train_df["dt"] = [datetime.datetime.fromtimestamp(time, datetime.timezone.utc) for time in train_df["TIMESTAMP"]]
# Time data

train_df["dt"].value_counts()
# Time data preparation

train_df["year"] = train_df["dt"].dt.year

train_df["month"] = train_df["dt"].dt.month

train_df["day"] = train_df["dt"].dt.day

train_df["hour"] = train_df["dt"].dt.hour

train_df["min"] = train_df["dt"].dt.minute

train_df["weekday"] = train_df["dt"].dt.weekday
train_df.head()
# Time series visualization

pivot = pd.pivot_table(train_df, index='month', columns="year", values="TRIP_ID", aggfunc="count").reset_index()



# Visualization, per month count

with plt.style.context("fivethirtyeight"):

    plt.figure(figsize=(10,6))

    plt.rcParams["font.size"] = 18

    plt.plot(pivot["month"], pivot[2013], label="2013")

    plt.plot(pivot["month"], pivot[2014], label="2014")

    plt.xlabel("month")

    plt.ylabel("count")

    plt.legend(facecolor="white")
# weekday, groupby whole data

weekday = pd.DataFrame(data=train_df.groupby("weekday").TRIP_ID.count()).reset_index()



with plt.style.context("fivethirtyeight"):

    plt.figure(figsize=(10,6))

    

    plt.plot(weekday["weekday"], weekday["TRIP_ID"])

    plt.xlabel("weekday\n (0:Monday ~ 6:Sunday)")

    plt.ylabel("count")

    plt.ylim([200000, 300000])
### Call type

call_type = pd.DataFrame(data=train_df.groupby("CALL_TYPE").TRIP_ID.count()).reset_index()



# visualization

with plt.style.context("fivethirtyeight"):

    plt.figure(figsize=(10,6))

    plt.bar(call_type["CALL_TYPE"], call_type["TRIP_ID"])

    plt.xlabel("CALL_TYPE")

    plt.ylabel("Count")
# 1st lon

lists_1st_lon = []

for i in range(0,len(train_df["POLYLINE"])):

    if train_df["POLYLINE"][i] == '[]':

        k=0

        lists_1st_lon.append(k)

    else:

        k = re.sub(r"[[|[|]|]|]]", "", train_df["POLYLINE"][i]).split(",")[0]

        lists_1st_lon.append(k)

        

train_df["lon_1st"] = lists_1st_lon



# 1st lat

lists_1st_lat = []

for i in range(0,len(train_df["POLYLINE"])):

    if train_df["POLYLINE"][i] == '[]':

        k=0

        lists_1st_lat.append(k)

    else:

        k = re.sub(r"[[|[|]|]|]]", "", train_df["POLYLINE"][i]).split(",")[1]

        lists_1st_lat.append(k)

        

train_df["lat_1st"] = lists_1st_lat
# last long

lists_last_lon = []

for i in range(0,len(train_df["POLYLINE"])):

        if train_df["POLYLINE"][i] == '[]':

            k=0

            lists_last_lon.append(k)

        else:

            k = re.sub(r"[[|[|]|]|]]", "", train_df["POLYLINE"][i]).split(",")[-2]

            lists_last_lon.append(k)



train_df["lon_last"] = lists_last_lon



# last lat

lists_last_lat = []

for i in range(0,len(train_df["POLYLINE"])):

    if train_df["POLYLINE"][i] == '[]':

        k=0

        lists_last_lat.append(k)

    else:

        k = re.sub(r"[[|[|]|]|]]", "", train_df["POLYLINE"][i]).split(",")[-1]

        lists_last_lat.append(k)

        

train_df["lat_last"] = lists_last_lat
# Delete lon & lat have "0".

train_df = train_df.query("lon_last != 0")
train_df["lon_1st"] = [float(k) for k in train_df["lon_1st"]]

train_df["lat_1st"] = [float(k) for k in train_df["lat_1st"]]

train_df["lon_last"] = [float(k) for k in train_df["lon_last"]]

train_df["lat_last"] = [float(k) for k in train_df["lat_last"]]
# Visualization, sampling 5000 datas.

mapping_1st = pd.DataFrame({

    "date":train_df.head(5000)["dt"].values,

    "lat":train_df.head(5000)["lat_1st"].values,

    "lon":train_df.head(5000)["lon_1st"].values

})



mapping_last = pd.DataFrame({

    "date":train_df.head(5000)["dt"].values,

    "lat":train_df.head(5000)["lat_last"].values,

    "lon":train_df.head(5000)["lon_last"].values

})



por_map = folium.Map(location=[41.141412,-8.590324], tiles='Stamen Terrain', zoom_start=11)



for i, r in mapping_1st.iterrows():

    folium.CircleMarker(location=[r["lat"],r["lon"]], radius=0.5, color="red").add_to(por_map)



for i, r in mapping_last.iterrows():

    folium.CircleMarker(location=[r["lat"],r["lon"]], radius=0.5, color="blue").add_to(por_map)    

    

por_map
train_df["delta_lon"] = train_df["lon_last"] - train_df["lon_1st"]

train_df["delta_lat"] = train_df["lat_last"] - train_df["lat_1st"]
# sampling : 5,000 point

sample = train_df.head(5000)



with plt.style.context("fivethirtyeight"):

    fig, ax = plt.subplots(1,3,figsize=(20, 6))

    

    # plot

    ax[0].scatter(sample["delta_lon"], sample["delta_lat"], s=3, c="red")

    ax[0].set_xlabel("delta longitude")

    ax[0].set_ylabel("delta_latitude")

    

    # delta longitude distribution

    ax[1].hist(sample["delta_lon"], bins=30, color="red")

    ax[1].set_xlabel("delta longitude")

    ax[1].set_ylabel("count")

    ax[1].set_yscale("log")

    

    # delta latitude distribution

    ax[2].hist(sample["delta_lat"], bins=30, color="red")

    ax[2].set_xlabel("delta latitude")

    ax[2].set_ylabel("count")

    ax[2].set_yscale("log")
# monthly, delta longtitude & latitude boxplot

train_df["month_str"] = [str(i) for i in train_df["month"]]
with plt.style.context("fivethirtyeight"):

    fig, ax = plt.subplots(1,2, figsize=(20,6))

    # delta_lon

    sns.boxplot("month_str", "delta_lon", data=train_df, ax=ax[0])

    ax[0].set_xlabel("month")

    ax[0].set_ylabel("delta_lon")

    ax[0].set_ylim([-0.3,0.3])

    # delta_lat

    sns.boxplot("month_str", "delta_lat", data=train_df, ax=ax[1])

    ax[1].set_xlabel("month")

    ax[1].set_ylabel("delta_lon")

    ax[1].set_ylim([-0.3,0.3])
# weekday, delta longtitude & latitude boxplot

train_df["weekday_str"] = [str(i) for i in train_df["weekday"]]
with plt.style.context("fivethirtyeight"):

    fig, ax = plt.subplots(1,2, figsize=(20,6))

    # delta_lon

    sns.boxplot("weekday_str", "delta_lon", data=train_df, ax=ax[0])

    ax[0].set_xlabel("weekday\n (0:Monday ~ 6:Sunday)")

    ax[0].set_ylabel("delta_lon")

    ax[0].set_ylim([-0.3,0.3])

    # delta_lat

    sns.boxplot("weekday_str", "delta_lat", data=train_df, ax=ax[1])

    ax[1].set_xlabel("weekday\n (0:Monday ~ 6:Sunday)")

    ax[1].set_ylabel("delta_lon")

    ax[1].set_ylim([-0.3,0.3])
# copy dataframe

df_ml = train_df.copy()



# outlier is dropped

df_ml = df_ml.query("delta_lon <= 0.2 & delta_lon >= -0.2 & delta_lat <= 0.2 & delta_lat >= -0.2")
# Call type <= 0.2 & delta_lon>=

map_call = {"A":1, "B":2, "C":3}

df_ml["Call_type"] = df_ml["CALL_TYPE"].map(map_call)



# Origin_call

def origin_call_flg(x):

    if x["ORIGIN_CALL"] == None:

        res = 0

    else:

        res = 1

    return res

df_ml["ORIGIN_CALL"] = df_ml.apply(origin_call_flg, axis=1)



# Origin_stand

def origin_stand_flg(x):

    if x["ORIGIN_STAND"] == None:

        res = 0

    else:

        res=1

    return res

df_ml["ORIGIN_STAND"] = df_ml.apply(origin_stand_flg, axis=1)



# Day type

df_ml.drop("DAY_TYPE", axis=1, inplace=True)



# Missing data

def miss_flg(x):

    if x["MISSING_DATA"] == "False":

        res = 0

    else:

        res = 1

    return res

df_ml["MISSING_DATA"] = df_ml.apply(miss_flg, axis=1)
df_ml = df_ml.sample(50000)
X = df_ml[["Call_type", 'ORIGIN_CALL', 'ORIGIN_STAND', 'MISSING_DATA', 'lon_1st', 'lat_1st', 'delta_lon', 'delta_lat']]
y = df_ml[["lon_last","lat_last"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
forest = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=1))



# Fitting

forest = forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)

y_test_pred = forest.predict(X_test)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred)))

print("MSE test;{}".format(mean_squared_error(y_test, y_test_pred)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred)))

print("R2 score test:{}".format(r2_score(y_test, y_test_pred)))
# 1st lon

lists_1st_lon = []

for i in range(0,len(test_df["POLYLINE"])):

    if train_df["POLYLINE"][i] == '[]':

        k=0

        lists_1st_lon.append(k)

    else:

        k = re.sub(r"[[|[|]|]|]]", "", test_df["POLYLINE"][i]).split(",")[0]

        lists_1st_lon.append(k)

        

test_df["lon_1st"] = lists_1st_lon



# 1st lat

lists_1st_lat = []

for i in range(0,len(test_df["POLYLINE"])):

    if test_df["POLYLINE"][i] == '[]':

        k=0

        lists_1st_lat.append(k)

    else:

        k = re.sub(r"[[|[|]|]|]]", "", test_df["POLYLINE"][i]).split(",")[1]

        lists_1st_lat.append(k)

        

test_df["lat_1st"] = lists_1st_lat
# last long

lists_last_lon = []

for i in range(0,len(test_df["POLYLINE"])):

        if test_df["POLYLINE"][i] == '[]':

            k=0

            lists_last_lon.append(k)

        else:

            k = re.sub(r"[[|[|]|]|]]", "", test_df["POLYLINE"][i]).split(",")[-2]

            lists_last_lon.append(k)



test_df["lon_last"] = lists_last_lon



# last lat

lists_last_lat = []

for i in range(0,len(test_df["POLYLINE"])):

    if test_df["POLYLINE"][i] == '[]':

        k=0

        lists_last_lat.append(k)

    else:

        k = re.sub(r"[[|[|]|]|]]", "", test_df["POLYLINE"][i]).split(",")[-1]

        lists_last_lat.append(k)

        

test_df["lat_last"] = lists_last_lat
# changin type str ⇒ float

test_df["lon_1st"] = [float(k) for k in test_df["lon_1st"]]

test_df["lat_1st"] = [float(k) for k in test_df["lat_1st"]]

test_df["lon_last"] = [float(k) for k in test_df["lon_last"]]

test_df["lat_last"] = [float(k) for k in test_df["lat_last"]]



# Create delta parameter

test_df["delta_lon"] = test_df["lon_last"] - test_df["lon_1st"]

test_df["delta_lat"] = test_df["lat_last"] - test_df["lat_1st"]
# copy dataframe

df_ml_t = test_df.copy()



# Call type <= 0.2 & delta_lon>=

map_call = {"A":1, "B":2, "C":3}

df_ml_t["Call_type"] = df_ml_t["CALL_TYPE"].map(map_call)



# Origin_call

df_ml_t["ORIGIN_CALL"] = df_ml_t.apply(origin_call_flg, axis=1)



# Origin_stand

df_ml_t["ORIGIN_STAND"] = df_ml_t.apply(origin_stand_flg, axis=1)



# Day type

df_ml_t.drop("DAY_TYPE", axis=1, inplace=True)



# Missing data

df_ml_t["MISSING_DATA"] = df_ml_t.apply(miss_flg, axis=1)
X_Test = df_ml_t[["Call_type", 'ORIGIN_CALL', 'ORIGIN_STAND', 'MISSING_DATA', 'lon_1st', 'lat_1st', 'delta_lon', 'delta_lat']]
y_Test_pred = forest.predict(X_Test)
submit_lat = y_Test_pred.T[1]

submit_lon = y_Test_pred.T[0]
submit = pd.DataFrame({"TRIP_ID":test_df["TRIP_ID"],

                     "LATITUDE":submit_lat,

                     "LONGITUDE":submit_lon})
submit.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")