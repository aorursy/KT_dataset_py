# Importing the necessary libraries.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# reading the data from the repository. 

location = "../input/covid19dataset/COVID_Data_Basic.csv"
data = pd.read_csv(location)
data
# Looking at the key features of our dataset.

data.describe()
#Checking the datatypes of each column.

data.dtypes
# Converting the datatype of Date to datatime format. 

data["Date"] = pd.to_datetime(data["Date"])
# Cross-checking whether we got the required datatype.

data.dtypes
# Checking for null values.

data.isnull().any()
# Checking for india's data. 

data["Country"].unique()
# Creating a dataframe with required features.

data_date = data.groupby("Date", as_index=False).Confirmed.sum()
data_date

# Looking at the datatypes.(in case they need to be astyped)

data_date.dtypes
# Creating a line-graph with above data.

plt.figure(figsize=(10,10))
sns.lineplot(x=data_date["Date"], y=data_date["Confirmed"])
plt.title("No of confirmed cases with time.")
plt.xticks(rotation=45)
# Creating a dataframe with the neccesary features/columns.

data_death = data.groupby("Country", as_index=False).Death.max()
data_death
# plotting a bargraph with the above dataframe. 

plt.figure(figsize=(15,40))
sns.barplot(x=data_death["Death"], y=data_death["Country"])
plt.yticks(size=10)
plt.xticks(size=20)
# Creating a dataframe of top 10 countries.

death = data_death.sort_values(by="Death", ascending=False)
death_top10 = death.head(10)
death_top10
# plotting a bargraph with the above dataframe.

plt.figure(figsize=(10,5))
sns.barplot(x=death_top10["Death"], y=death_top10["Country"])
plt.title("top countries with the most death cases")
# Creating a dataframe of countries vs recovered cases.

data_recover = data.groupby("Country", as_index=False).Recovered.max()
data_recover
# Creating a dataframe of countries vs confirmed cases.

data_confirmed = data.groupby("Country", as_index=False).Confirmed.max()
data_confirmed
# Merging the above 2 Dataframes.
data_fight = pd.merge(data_recover, data_confirmed, on='Country')
data_fight
# Creating the ratios by dividing confirmed cases/ recovered cases.

ratio = data_fight["Confirmed"]/data_fight["Recovered"]
ratio
# Adding the above ration to the existing dataframe for easier evaluation.

data_fight["Ratio"] = ratio
data_fight
# checking the datatype of ratio column. 

data_fight.dtypes
# plotting a bargraph with the above dataframe.

plt.figure(figsize=(15,40))
sns.barplot(x=data_fight["Ratio"], y=data_fight["Country"])
plt.yticks(size=10)
plt.xticks(size=20)
# sorting the dataframe with the least ratios.

data_top10 = data_fight.sort_values(by="Ratio", ascending=True)
data_top10.head(10)
# Here, we need to drop "Diamond Princess" beacause that's a cruising ship and and not a country. 

data_top10 = data_top10.drop(index=[49])
data_top10 = data_top10.head(10)
plt.figure(figsize=(10,10))
sns.barplot(x=data_top10["Country"], y=data_top10["Ratio"])
plt.title("Countries vs ratio of confirmed cases to recoveries")
# Extracting country specific data.

data_india = data[data['Country']=="India"]
data_india
# Confirmed cases in India in each day from 2019-12-31 to 2020-4-22.

india_confirmed = data_india.groupby("Date", as_index=False).Confirmed.sum()
india_confirmed
# Plotting the above dataframe in a line graph.

plt.figure(figsize=(10,10))
sns.lineplot(x=india_confirmed["Date"], y=india_confirmed["Confirmed"])
plt.xticks(rotation=45)
plt.title("India's rising covid-19 cases")
# # Recovered cases in India in each day from 2019-12-31 to 2020-4-22.

india_recover = data_india.groupby("Date", as_index=False).Recovered.sum()
india_recover
india_death = data_india.groupby("Date", as_index=False).Death.sum()
india_death
# plotting the above dataframes in a line graph. 

plt.figure(figsize=(10,10))
sns.lineplot(x=india_recover["Date"], y=india_recover["Recovered"], label="recovered cases")
sns.lineplot(x=india_confirmed["Date"], y=india_confirmed["Confirmed"], label="confirmed cases")
sns.lineplot(x=india_death["Date"], y=india_death["Death"], label="Death cases")
plt.xticks(rotation=45)
plt.ylabel("no of cases")