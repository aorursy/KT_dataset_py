# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.figure_factory as ff
from tqdm import tqdm
import plotly.offline as py
import plotly.express as px
import seaborn as sns
import random
from collections import Counter
import warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
cars_df = pd.read_csv("/kaggle/input/personal-cars-classifieds/all_anonymized_2015_11_2017_03.csv")
cars_df.columns
cars_df.head(10)
cars_df.describe()
cars_df.drop(labels = ["body_type","color_slug","stk_year","date_created","date_last_seen"],axis=1,inplace=True)

print(cars_df[cars_df["fuel_type"].isnull()])
print("*"*40)
print(cars_df[cars_df["fuel_type"]== "electric"])
print("*"*40)
print(cars_df[cars_df["fuel_type"]== "lpg"])
print("*"*40)
print(cars_df[cars_df["fuel_type"]== "cng"])
print("*"*40)
cars_df.drop(cars_df[cars_df["fuel_type"].isnull()].index,axis=0,inplace=True)
cars_df.drop(cars_df[cars_df["fuel_type"]== "electric"].index,axis=0,inplace=True)
cars_df.drop(cars_df[cars_df["fuel_type"]== "lpg"].index,axis=0,inplace=True)
cars_df.drop(cars_df[cars_df["fuel_type"]== "cng"].index,axis=0,inplace=True)
fuel=cars_df["fuel_type"]
cars_df["fuel"] = [1 if i=="gasoline" else 0 for i in fuel]
# diesel 0
# gasoline 1
cars_df.columns[cars_df.isnull().any()]
cars_df.isnull().sum()
cars_df.info()
def pie_plot(variable):
    """
        input: variable ex: "Model"
        output: pie plot & value count
    """
    #get feature
    var = cars_df[variable]
    #count number of categorical variable(value/sample)
    varValue = var.value_counts()
    #visualize
    fig = px.pie(values=varValue,names=varValue.index,template="plotly_white",title="Cars Pie Plot in {} column".format(variable))
    fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
    fig.show()
    print("{}:\n{}".format(variable,varValue))
categoricals = ["maker","model","transmission","door_count","seat_count","fuel_type"]
for c in categoricals:
    pie_plot(c)
# Maker vs fuel
cars_df[["maker","fuel"]].groupby(["maker"], as_index= False).mean().sort_values(by="fuel",ascending= False)
# model vs fuel
pd.set_option('display.max_rows',1000) #we see all features
cars_df[["model","fuel"]].groupby(["model"], as_index= False).mean().sort_values(by="fuel",ascending= False)
# Transmision vs fuel
cars_df[["transmission","fuel"]].groupby(["transmission"], as_index= False).mean().sort_values(by="fuel",ascending= False)
# door_count vs fuel
cars_df[["door_count","fuel"]].groupby(["door_count"], as_index= False).mean().sort_values(by="fuel",ascending= False)
# seat_count vs fuel
cars_df[["seat_count","fuel"]].groupby(["seat_count"], as_index= False).mean().sort_values(by="fuel",ascending= False)
# fuel_type vs fuel
cars_df[["fuel_type","fuel"]].groupby(["fuel_type"], as_index= False).mean().sort_values(by="fuel",ascending= False)
def detect_outliers(df,features):
    outlier_indices = []
    for c in features:
        #1st Quartile
        Q1 = np.percentile(df[c],25)
        #3rd Quartile
        Q3 = np.percentile(df[c],75)
        #IQR
        IQR = Q3-Q1
        #Outlier Step
        outlier_step = IQR*1.5
        #Detect Outlier and their Indeces
        outlier_list_col = df[(df[c]< Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # Store indeces
        outlier_indices.extend(outlier_list_col),
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i,v in outlier_indices.items() if v > 2)
    return multiple_outliers
cars_df.loc[detect_outliers(cars_df,["engine_displacement","engine_power","price_eur","mileage","manufacture_year"])]
cars_df = cars_df.drop(detect_outliers(cars_df,["engine_displacement","engine_power","price_eur","mileage","manufacture_year"]), axis=0).reset_index(drop = True)
cars_df.columns[cars_df.isnull().any()]
cars_df.isnull().sum()
#pd.set_option('display.max_rows',421755) if you want to  check all of them is NaN with your eyes open this code and delete last .isnull().all()
cars_df["model"][cars_df["maker"].isnull()].isnull().all()
cars_df[cars_df["maker"].isnull()]
cars_df.drop(cars_df[cars_df["maker"].isnull()].index,axis=0,inplace=True)
cars_df.isnull().sum()
cars_df["maker"][cars_df["model"].isnull()].value_counts()
cars_df["model"][(cars_df.maker=="infinity") & (cars_df.fuel==1)].value_counts()
cars_df["model"][(cars_df.maker=="infinity") & (cars_df.model.isnull())].index
cars_df["model"][3393719] = "fx35"
cars_df[(cars_df.maker=="aston-martin")]
cars_df["model"][3350774] = "rapide"
cars_df["model"][3350765] = "rapide"
cars_df["model"][3350756] = "rapide"
cars_df["model"][3350747] = "v8-vantage"
cars_df[(cars_df.maker=="aston-martin")]
index_nan_model = list(cars_df["model"][cars_df["model"].isnull()].index)
model_med = cars_df["model"].value_counts().idxmax()
model_med
def fill_nulls(fill_column,other_column1,other_column2,other_column3,idx):
    pred = cars_df[fill_column][(( cars_df[other_column1] == cars_df.iloc[idx][other_column1]) &
                             ( cars_df[other_column2] == cars_df.iloc[idx][other_column2]) &
                             ( cars_df[other_column3] == cars_df.iloc[idx][other_column3]))].value_counts()

    if not pred.empty:
        print(idx,"degisti")
        cars_df["model"].iloc[idx]= pred.idxmax()
    else:
        print(idx,"silindi")
        cars_df.drop(idx,axis=0,inplace=True)
"""
for i in tqdm(index_nan_model:
    fill_nulls("model","maker","fuel","manufacture_year",i)
"""
cars_df.to_csv("temizleme2.csv",index=False)
cars_df = pd.read_csv("/kaggle/input/temizleme/temizleme2.csv",low_memory=False)
cars_df.isnull().sum()
cars_df.drop(cars_df[cars_df["engine_displacement"].isnull()].index,axis=0,inplace=True)
cars_df.drop(cars_df[cars_df["engine_power"].isnull()].index,axis=0,inplace=True)
cars_df.drop(cars_df[cars_df["transmission"].isnull()].index,axis=0,inplace=True)
cars_df.drop(cars_df[cars_df["door_count"].isnull()].index,axis=0,inplace=True)
cars_df.drop(cars_df[cars_df["seat_count"].isnull()].index,axis=0,inplace=True)
cars_df.isnull().sum()
cars_df.to_csv("temizleme3.csv",index=False)
cars_df[cars_df["mileage"].isnull()]
cars_df["model"][cars_df["mileage"].isnull()].value_counts()
cars_df["mileage"][(cars_df.model=="g400-cdi") & (cars_df.mileage.isnull())].index
cars_df[(cars_df.model=="g400-cdi") & (cars_df.manufacture_year <= 1990)]
"""
from tqdm import tqdm
index_nan_mileage = list(cars_df["mileage"][cars_df["mileage"].isnull()].index)
mileage_med = cars_df["mileage"].median()
for i in tqdm(index_nan_mileage):
    mileage_pred = cars_df["mileage"][((cars_df["manufacture_year"] == cars_df.iloc[i]["manufacture_year"]) &(cars_df["model"] == cars_df.iloc[i]["model"]))].median()
    if not np.isnan(mileage_pred):
        print("Change value")
        cars_df["mileage"].iloc[i] = mileage_pred   
    else:
        print("Average value")
        cars_df["mileage"].iloc[i] = mileage_med
"""
cars_df.to_csv("temizleme4.csv",index=False)
"""
from tqdm import tqdm
index_nan_manufacture_year = list(cars_df["manufacture_year"][cars_df["manufacture_year"].isnull()].index)
manufacture_year_med = cars_df["manufacture_year"].median()
for i in tqdm(index_nan_manufacture_year):
    manufacture_year_pred = cars_df["manufacture_year"][((cars_df["model"] == cars_df.iloc[i]["model"])
                                                         & (cars_df["mileage"] == cars_df.iloc[i]["mileage"])
                                                        )].median()
    if not np.isnan(manufacture_year_pred):
        print("Change value")
        cars_df["mileage"].iloc[i] = manufacture_year_pred   
    else:
        print("Average value")
        cars_df["mileage"].iloc[i] = manufacture_year_med
"""
cars_df = pd.read_csv("/kaggle/input/temizleme/temizleme4.csv",low_memory=False)
cars_df.isnull().sum()