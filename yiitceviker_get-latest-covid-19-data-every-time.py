## GET DAILY COVID 19 ONLINE DATA FROM https://www.worldometers.info/coronavirus/ AND 
## ADD TO novel-corona-virus DATASET SO YOU CAN LOOK LATEST DATA AND GRAPHICS EVERY TIME

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
from bs4 import BeautifulSoup
%matplotlib inline
plt.rcParams['figure.figsize'] = 17,15
#Select Countries
countries = ["United Kingdom","Thailand","Italy","Spain","Germany","Switzerland","Iran","US","France","Turkey"]
## GET ONLINE DATA
source_code = requests.get('https://www.worldometers.info/coronavirus/')
soup = BeautifulSoup(source_code.text)
table = soup.find_all('table')[0] 
data = []
rows = table.find_all('tr')[1:]
## ONLINE DATA PARSE
def getOnlineData(countriesData):
    st = pd.DataFrame(index=countriesData)
    cconfirm = []
    cdeaths = []
    for country in countriesData:
        if country == "United Kingdom":
            country = "UK"
        remain = -1
        for row in rows:
            cols = row.find_all('td')
            for col in cols:
                if remain >= 0:
                    remain+=1
                if remain == 1:
                    cconfirm.append (int(col.text.replace(",","")))
                if remain == 3:
                    cdeaths.append(int(col.text.replace(",","")))
                    remain = -1    
                if col.text.find(country) == 0:
                    remain = 0
  
    st["comfirms"] = cconfirm
    st["deaths"] = cdeaths
    return st  

onlineData = getOnlineData(countries)
##Confirmed
confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv" )
confirmed.rename(columns={"Province/State":"State", "Country/Region":"Country"}, inplace=True)
confirmed.set_index("Country", inplace=True)
confirmed.drop(columns=["State","Lat","Long"],inplace=True)
confirmed.sort_index()
confirmed.columns = pd.to_datetime(confirmed.columns)
confirmed = confirmed.loc[countries]
##Deaths
deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv" )
deaths.rename(columns={"Province/State":"State", "Country/Region":"Country"}, inplace=True)
deaths.set_index("Country", inplace=True)
deaths.drop(columns=["State","Lat","Long"],inplace=True)
deaths.sort_index()
deaths.columns = pd.to_datetime(deaths.columns)
deaths = deaths.loc[countries]
confirmed = confirmed.groupby(confirmed.index).sum()
deaths = deaths.groupby(deaths.index).sum()
## ADD ONLINE DATA TO DATASET
from datetime import date
today = pd.to_datetime(date.today())
confirmed[today] = onlineData["comfirms"]
deaths[today] = onlineData["deaths"]
confirmed
#deaths
## DAILY DIFFERENCE
confTrans = confirmed.transpose()
deathsTrans = deaths.transpose()
confDiff = pd.DataFrame()
for col in confTrans.columns:
    confDiff[col] = confTrans[col] - confTrans[col].shift(1)
deathsDiff = pd.DataFrame()
for col in deathsTrans.columns:
    deathsDiff[col] = deathsTrans[col] - deathsTrans[col].shift(1)
#Add for bypass zero values
addTrans = 45
confTrans.iloc[addTrans:len(confTrans)].plot(title="VTotal Confirms")
deathsTrans[addTrans: len(deathsTrans)].plot(title="Total Deaths")
#Add for bypass zero values
addDiff = 40
confDiff[addDiff:len(confDiff)].plot(title="Daily Difference of Confirmed", kind ="line" )
confDiff[addDiff:len(confDiff)].plot(title="Daily Difference of Confirmed", kind ="box" )
deathsDiff[addDiff: len(deathsDiff)].plot(title="Daily Difference of Deaths")
deathsDiff
deathsDiff[addDiff: len(deathsDiff)].plot(title="Daily Difference of Deaths", kind = "box")