import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np

print("Setup Complete")
crimes_filepath = "../input/crimes-in-boston/crime.csv"



crimes_data = pd.read_csv(crimes_filepath, index_col=0, encoding="latin-1")



crimes_data.head()
pd.pivot_table(crimes_data, index=["DISTRICT", "MONTH"])
heatmap_data = crimes_data.filter(['MONTH','DAY_OF_WEEK'],axis=1)

heatmap_data.head()
heatmap_pivot_table = pd.pivot_table(heatmap_data, index=["MONTH"], columns=["DAY_OF_WEEK"], aggfunc=[len])

heatmap_pivot_table.head()
plt.figure(figsize=(14,7))

plt.title("Total amount of crime by days of the week")

sns.heatmap(data=heatmap_pivot_table)
swarmplot_data = crimes_data.filter(['OFFENSE_CODE_GROUP','OCCURRED_ON_DATE'], axis=1)

swarmplot_data.head()

swarmplot_data['OCCURRED_ON_DATE'] = swarmplot_data['OCCURRED_ON_DATE'].str.slice(10, 13, 1)

swarmplot_data['OCCURRED_ON_DATE'].head() 
swarmplot_data['OCCURRED_ON_DATE'] = swarmplot_data['OCCURRED_ON_DATE'].astype(int)

swarmplot_data.head() 
plt.figure(figsize=(75,100))

plt.title("Crimes in Boston")

plt.yticks(rotation=45)

sns.set(font_scale=5) 

sns.swarmplot(y=swarmplot_data['OFFENSE_CODE_GROUP'].head(2000), x=swarmplot_data['OCCURRED_ON_DATE'].head(2000), size=10)
crime_type_data = crimes_data.filter(['OFFENSE_CODE_GROUP','OCCURRED_ON_DATE'], axis=1)

crime_type_data.head()
crime_type_data['OCCURRED_ON_DATE'] = crime_type_data['OCCURRED_ON_DATE'].str.slice(10, 13, 1)

crime_type_data.head()
crime_type_data['OCCURRED_ON_DATE'] = crime_type_data['OCCURRED_ON_DATE'].astype(int)

crime_type_data.head()
crime_type_pivot_table = pd.pivot_table(crime_type_data, index=["OFFENSE_CODE_GROUP"], columns=["OCCURRED_ON_DATE"], aggfunc=[len])

crime_type_pivot_table.head()
plt.figure(figsize=(30,30))

plt.title("Crime Time")

sns.heatmap(data=crime_type_pivot_table)