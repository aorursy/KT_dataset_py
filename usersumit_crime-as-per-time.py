import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
# crimes data

data_crimes = pd.read_csv('../input/crimes.csv', header=0)



data_crimes['ReportedDate'] = data_crimes['ReportedDate'].apply(lambda x: str(x)[0:4])

data_crimes['Time'] = data_crimes['Time'].apply(lambda x: str(x)[0:2])



data_crimes=data_crimes.rename(columns = {'ReportedDate':'Year'})

data_crimes["count"] = 1

data_crimes.head()
dropColumns = ['publicaddress', 'controlnbr', 'CCN', 'Precinct', 'BeginDate', 'UCRCode', 'EnteredDate', 'Long', 'Lat', 'x', 'y','lastchanged','LastUpdateDate','OBJECTID','ESRI_OID']

data_crimes = data_crimes.drop(dropColumns, axis=1)

data_crimes.head()
data_crimes_per_year = data_crimes.groupby(["Year", "Time"]).sum().reset_index()

data_crimes_per_year['count'] = data_crimes_per_year['count'].apply(lambda x: (x + 1))

data_crimes_per_year.head()
data_crimes_per_year.reset_index().pivot_table(index="Year", columns="Time", values="count").T.plot(kind='barh', stacked=True, figsize=(10, 14))