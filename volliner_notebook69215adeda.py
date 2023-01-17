import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

import seaborn as sns
ap = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv")

ap.head(3)
ap = ap.assign(Year = pd.to_datetime(ap["Date"]).dt.year)

ap = ap.assign(People_Alive = ap["Aboard"]-ap["Fatalities"])

ap = ap.assign(Alive_Percent = (ap["People_Alive"]/ap["Aboard"])*100.0)
table_count = ap.groupby([ap['Year']])['Fatalities'].size()



year = table_count.index

table_count_val = table_count.values

fig,ax = plt.subplots(figsize=(15,6))

sns.barplot(x = year , y = table_count_val)

plt.title('Fatalities per year')

plt.xlabel('Year')

plt.ylabel('Count')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
ap.plot.scatter("Aboard", "Fatalities", alpha=0.2, figsize=(12,5), color="purple");
ap.Year = pd.to_numeric(ap.Year, errors='coerce').fillna(0).astype(np.int64)

ap = ap[ap["Year"] > 1945]

ap1 = ap.groupby("Year")[["Aboard", "Fatalities", "People_Alive"]].sum()

ap3 = ap.groupby("Year")[["People_Alive"]].sum()

ap4 = ap.groupby("Year")[["Aboard"]].sum()

ap5 = ap3["People_Alive"] / ap4["Aboard"]

ap5.columns = ["Alive_Percent"]

ap6 = pd.concat([ap1, ap5], axis=1)

ap6.columns = ['Aboard', 'Fatalities', 'People_Alive', 'Alive_Percent']

ap7 = ap6.sort_values("Alive_Percent", ascending=False).head(30)



cm = sns.light_palette("lightblue", as_cmap=True)

s = ap7.style.background_gradient(cmap=cm)

s