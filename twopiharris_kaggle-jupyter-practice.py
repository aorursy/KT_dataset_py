#upper = int(input("upper limit? "))

upper = 5

for i in range(upper):

    print(i)
#loading in standard libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting library
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print("files available: ")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#create a data file to play with

names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']

births = [968, 155, 77, 578, 973]



BabyDataSet = list(zip(names, births))

print(BabyDataSet)



df = pd.DataFrame(BabyDataSet, columns = ("Names", "Count"))

df
df = df.set_index("Names")

df.plot(kind = "bar")
#read from sqlite database

import sqlite3

conn = sqlite3.connect("../input/database.sqlite")
#get table names

c = conn.cursor()

result = c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")

for row in result:

    print (row[0])
# get column names for indicators table

df = pd.read_sql_query("SELECT * FROM Indicators LIMIT 1", conn)

print(df.columns.values)



df = pd.read_sql_query("SELECT * FROM Indicators LIMIT 20", conn)

df
#opening file in pandas

query = """

SELECT

  Year as 'Year',

  Value as 'Military exports'

FROM 

  indicators

WHERE

  indicatorCode= 'MS.MIL.XPRT.KD'

AND 

  CountryCode = "ARB"

"""

df = pd.read_sql_query(query, conn)

df
df = df.set_index("Year")

df.plot(figsize = (12,8), kind = "bar")
df = pd.read_csv("../input/Country.csv")

#select name of element 10

print(df.loc[10,['LongName']])





# select countrycode and shortname from first ten elements

countryName =  df.loc[0:10,['CountryCode', 'ShortName']]

countryName
df = pd.read_csv("../input/Indicators.csv")



#get the results from Afghanistan

afg = df[df.CountryCode == "AFG"]

afg.head()
#get members of afg dataset from 1960

afg60 = afg[df.Year == 1960]

afg60
#remove index issue by combining with boolean operators

#get all population values for Afghanistan

afgPop = df[(df.CountryCode == "AFG") & (df.IndicatorCode == "SP.POP.TOTL")]

afgPop
afgPop = afgPop[["Year", "Value"]]

afgPop = afgPop.set_index("Year")

afgPop.plot()