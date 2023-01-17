import numpy as np

import pandas as pd

from sklearn import linear_model



def get_training_data():

    df = pd.read_csv("../input/testset.csv")

    #pre processed training data

    df = df.loc[:, [' _hum',' _tempm',' _rain']]

    df = df.dropna()

    return df

    

def train_model(df):

    reg = linear_model.LinearRegression()

    print("** Training dataset!! Please wait!! **")

    #print(df[' _rain'])

    reg.fit(df[[' _hum', ' _tempm']], df[' _rain'])

    return reg

    

def prediction(temp, humidity):

    df = get_training_data()

    model = train_model(df)

    rain_possiblity = model.predict([[humidity,temp]])

    return get_assumption(rain_possiblity[0])



def get_assumption(val):

    if (val > 0.07):

        return "High Possiblity of Rain!!"

    elif (0.07 > val > 0.04):

        return "Possiblity of Rain Today!!"

    elif (0.04 > val > -0.01):

        return "Low Possiblity of Rain Today!!"

    elif (val < -0.01):

        return "No!! Possiblity of Rain Today!!"

    

def entry():

    temp = float(input("Today's Temperature in C : "))

    humidity = float(input("Today's Humidity in % : "))

    return temp, humidity



print("--- Prediction of Rain ---")

temp, humidity = entry()

prediction(temp, humidity)