

import pandas as pd

import numpy as np

import requests

from bs4 import BeautifulSoup



Url = "https://www.worldometers.info/coronavirus/"



page = requests.get(Url).text







soup = BeautifulSoup(page,"lxml")





get_table = soup.find("table",id="main_table_countries_today")





table_data = get_table.tbody.find_all("tr")

print(table_data)



table_columns = get_table.thead.find_all("th")



print(len(table_columns))



columns = []

for i in range(len(table_columns)):

   l = table_columns[i].text

   columns.append(l) 



"""

for i in range(len(table_data)):

   key = table_data[i].find_all("a",href=True)

   print(key)  """



Dict = {}

for i in range(len(table_data)):

    try:

      key = (table_data[i].find_all('a', href=True)[0].string)

    except:

      key = (table_data[i].find_all('td')[0].string)

      

    value = [j.string for j in table_data[i].find_all("td")[1:]]  

    Dict[key] = value

print(Dict)    



live_data= pd.DataFrame(Dict).T.iloc[:,:12]

live_data.head()

df5 = live_data.reset_index()

df5.head()

df5.columns = columns

df5.head()

df6 = df5.iloc[2:,:]

df7 = df6.reset_index(drop=True)

df7.head()

df7.columns =  ['Country_name', 'TotalCases', 'NewCases', 'TotalDeaths', 'NewDeaths',

       'TotalRecovered', 'ActiveCases', 'Serious,Critical', 'Tot_Cases/1M_pop',

       'Deaths/1M_pop', 'TotalTests', 'Tests/1M_pop', 'Continent']

df7.to_csv("CoronaLive1.csv",index=False)
df7.head()