#Import libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

#Load data from csv

rawData = pd.read_csv("../input/ufo_sighting_data.csv")

#Print first few rows

rawData.head()

def getMonth(dateTime):

    end = dateTime.index('/')

    return int(dateTime[:end])





def getDay(dateTime):

    begin = dateTime.index('/') + 1

    end = dateTime.index('/', begin)

    return int(dateTime[begin : end])





def getYear(dateTime):

    begin = dateTime.index('/', 3) + 1

    end = dateTime.index(' ', begin)

    return int(dateTime[begin : end])





#Feature engineering

#Parse date

rawData["Month"] = rawData["Date_time"].apply(lambda x: getMonth(x))

rawData["Day"] = rawData["Date_time"].apply(lambda x: getDay(x))

rawData["Year"] = rawData["Date_time"].apply(lambda x: getYear(x))

#get all sightings where the country is US

usData = rawData[rawData["country"].isin(["us"])]



#Find sightings with missing country

nanData = rawData[rawData["country"].apply(pd.isnull)]



#get sigtings in US states

usStates = usData["state/province"].unique()

nanUSData = nanData[nanData["state/province"].isin(usStates)]

nanUSData = nanUSData.fillna({"country":"us"})



#add to usData

usData = usData.append(nanUSData)

usData["country"].unique()
sightingsByState = usData.groupby("state/province").size()

usSightingMean = sightingsByState.mean()

print("Sighting per State mean = " + str(usSightingMean))

topStates = sightingsByState.where(lambda x : x >= usSightingMean * 2).dropna()

print("States in > mean = " + str(topStates))
#set index to datetime

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

for state in topStates.index:

    stateSightings = usData[usData["state/province"] == state]

    x = stateSightings.groupby("Year").size()

    x.plot(kind="line", title="Sightings")


