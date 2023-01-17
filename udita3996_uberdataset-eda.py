import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

uber_dataset = pd.read_csv("/kaggle/input/uber-request-data/Uber Request Data.csv")

uber_dataset.head()
uber_dataset.shape
#Check the datatypes of column

uber_dataset.info()
#Convert Request_timestamp to uniform datetime format

uber_dataset["Request timestamp"] = uber_dataset["Request timestamp"].apply(lambda x : pd.to_datetime(x))

uber_dataset.info()
#Convert drop_timestamp to uniform datetime format

uber_dataset["Drop timestamp"] = uber_dataset["Drop timestamp"].apply(lambda x : pd.to_datetime(x))

uber_dataset.info()
#Check for null values

uber_dataset.isnull().sum()
uber_dataset.Status.value_counts()
#Check if the Driver id is null only for 'No Cars Available' Status

uber_dataset[(uber_dataset.Status == 'No Cars Available') & (uber_dataset["Driver id"].isnull())].shape
#Check if drop timestamp is null only for 'No Cars Available' & 'Cancelled'

uber_dataset[((uber_dataset.Status == 'No Cars Available') | (uber_dataset.Status == 'Cancelled')) & (uber_dataset["Drop timestamp"].isnull())].shape
#Extract the hour from requested timestamp

uber_dataset["Request hour"] = uber_dataset["Request timestamp"].dt.hour

uber_dataset.head(5)
plt.hist(uber_dataset["Request hour"],edgecolor='black',bins=24)

plt.xlabel("Request hour")

plt.ylabel("No. of Requests")

plt.show()

#Demand is more during evening hours
#divide the time of the day into five categories

def time_period(x):

    if x < 5:

        return "Early Morning"

    elif 5 <= x < 10:

        return "Morning"

    elif 10 <= x < 17:

        return "Day Time"

    elif 17 <= x < 22:

        return "Evening"

    else:

        return "Late Night"



    

uber_dataset['Time slot'] = uber_dataset['Request hour'].apply(lambda x: time_period(x))

uber_dataset['Time slot'].value_counts().plot.bar()

plt.show()

#Maximum demand during 'Evening' hours
uber_dataset["Pickup point"].value_counts().plot.pie(autopct='%1.0f%%')

plt.show()
uber_dataset["Status"].value_counts().plot.pie(autopct='%1.0f%%')

plt.show()

#More than half of the requests are either cancelled or on wait due to unavailability of cabs
uber_dataset["Count"] = 1

uber_city = uber_dataset[uber_dataset["Pickup point"]=="City"]

uber_airport = uber_dataset[uber_dataset["Pickup point"]=="Airport"]
#Availability matrix for requests with Pickup point as City

pivot_city = pd.pivot_table(uber_city,index = "Time slot",columns = "Status",values = "Count",aggfunc=np.sum)

plt.figure(figsize = [8,6])

hm = sns.heatmap(data = pivot_city, annot = True, fmt='g')

plt.show()
#Availability matrix for requests with Pickup point as Airport

pivot_airport = pd.pivot_table(uber_airport,index = "Time slot",columns = "Status",values = "Count",aggfunc=np.sum)

plt.figure(figsize = [8,6])

hm = sns.heatmap(data = pivot_airport, annot = True, fmt='g')

plt.show()