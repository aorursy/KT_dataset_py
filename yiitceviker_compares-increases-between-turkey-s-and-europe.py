## Covid 19 virus has arrived to Turkey after a month of European countries. This sheet shows ,compares increases of Covid 19 between Turkey's and European Countries.
## Time shifting and editing were made to match the data.
##  Joined https://www.worldometers.info/coronavirus/ data and Covid 19 dataset to get the latest data every time

## Türkiye'ye COVID 19 gelmesinden itibaren, salgının diğer Avrupa ülkelerindeki başlangıç hızı ile karşılaştırılması.

import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
%matplotlib inline

plt.rcParams['figure.figsize'] = 17,11
st = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv", squeeze=True)
st.columns = ['SNo', 'Date', 'City', 'Country','Update', 'Confirmed', 'Deaths', 'Recovered']

countries = ["Italy","Spain","Germany", "Turkey","Switzerland"]
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
def countryData(country, scrollNum):
    tempC = st[st.Country == country]
    maskC = ~tempC["Confirmed"].duplicated()
    tempC = tempC[maskC]
    couTemp = tempC[["Date","Confirmed","Deaths"]]
    couTemp.set_index("Date", inplace=True)
    cou = couTemp[scrollNum : len(turkey) + scrollNum]
    couConfirmed = couTemp.iloc[len(turkey) + scrollNum]["Confirmed"]
    couDeaths = couTemp.iloc[len(turkey) + scrollNum]["Deaths"]
    return [cou, couConfirmed, couDeaths]
#Turkey
turkey = st[st.Country == 'Turkey'][["Date","Confirmed","Deaths"]]
#Italy
italy, italyConfirmed, italyDeaths = countryData("Italy", 0)
#Spain
spain, spainConfirmed, spainDeaths = countryData("Spain", 3)
#Germany
germany, germanyConfirmed, germanyDeaths = countryData("Germany", 10)
#Switzerland
switzerland, switConfirmed, switDeaths = countryData("Switzerland", 4)
#
confirmed = pd.DataFrame (turkey["Confirmed"].values, index=range(1,len(turkey)+1), columns=["Turkey"])
confirmed["Italy"] = italy["Confirmed"].values
confirmed["Spain"] = spain["Confirmed"].values
confirmed["Germany"] = germany["Confirmed"].values
confirmed["Switzerland"] = switzerland["Confirmed"].values

deaths = pd.DataFrame (turkey["Deaths"].values, index=range(1,len(turkey)+1), columns=["Turkey"])
deaths["Italy"] = italy["Deaths"].values
deaths["spain"] = spain["Deaths"].values
deaths["Germany"] = germany["Deaths"].values
deaths["Switzerland"] = switzerland["Deaths"].values
##EKLEME
from datetime import date
tConfirmed = onlineData.loc["Turkey"]["comfirms"]
tDeaths = onlineData.loc["Turkey"]["deaths"]
today = date.today().strftime("%m/%d/%Y")
confirmed.loc[len(confirmed) +1] = [tConfirmed, italyConfirmed, spainConfirmed, germanyConfirmed,switConfirmed]
deaths.loc[len(deaths) +1] = [tDeaths, italyDeaths, spainDeaths, germanyDeaths, switDeaths]
confirmed.plot(title="Compares comfirmed increases of Covid 19 between Turkey's and European Countries")
deaths.plot(title = "Compares deaths increases of Covid 19 between Turkey's and European Countries")
confirmed.tail(3)
fig, ax1 = plt.subplots(figsize = (14, 5))

spa = ax1.bar(confirmed.index, confirmed.Spain, bottom = 0,  label = "Spain", color = "Green")
ger = ax1.bar(confirmed.index, confirmed.Germany ,  label = "Germany", color = "Orange")
ita = ax1.bar(confirmed.index, confirmed.Italy ,  label = "Italy", color = "Blue")
tur = ax1.bar(confirmed.index, confirmed.Turkey,  label = "Turkey", color = "Red")
tur = ax1.bar(confirmed.index, confirmed.Switzerland,  label = "Switzerland", color = "Purple")

ax1.legend()
ax1.set_title("Compares comfirmed increases of Covid 19 between Turkey's and European Countries", size = 22)
ax1.set_xlabel("Days", size = 18, color = "r")
ax1.set_ylabel("Deaths", size = 18, color = "r")
fig, ax = plt.subplots(figsize = (14, 5))

spa = ax.bar(deaths.index, deaths.spain, bottom = 0,  label = "Spain", color = "Green")
ita = ax.bar(deaths.index, deaths.Italy ,  label = "Italy", color = "Blue")
tur = ax.bar(deaths.index, deaths.Turkey,  label = "Turkey", color = "Red")
ger = ax.bar(deaths.index, deaths.Germany ,  label = "Germany", color = "Orange")
tur = ax.bar(deaths.index, deaths.Switzerland,  label = "Switzerland", color = "Purple")

ax.legend()
ax.set_title("Compares deaths increases of Covid 19 between Turkey's and European Countries", size = 22)
ax.set_xlabel("Days", size = 18, color = "r")
ax.set_ylabel("Deaths", size = 18, color = "r")
