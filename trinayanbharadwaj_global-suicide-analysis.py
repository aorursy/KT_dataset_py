#importing the necessary libraries.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#reading the dataset from its directory.

location = "../input/suicide-rates-overview-1985-to-2016/master.csv"
df = pd.read_csv(location)
#showing the first 5 rows of the dataframe. 

df.head()
#checking if the dataset requires some cleaning (fixing trash data).

df.isnull().any()
df = df.drop(columns="HDI for year")
df.dtypes
df.columns

df = df.drop(columns=' gdp_for_year ($) ')
#Creating a heatmap of the dataset.

plt.figure(figsize=(15,10))
cor = df.corr()
sns.heatmap(data=cor, annot=True)
plt.yticks(rotation=0)
#Here we create a separate dataframe of "age" and "suicides_no".

data_age = df.groupby("age", as_index=False).suicides_no.sum()
data_age
#Here we plot the above dataframe. 

plt.figure(figsize=(10,5))
sns.barplot(x=data_age["suicides_no"], y=data_age["age"])
plt.title("showing the relation between age group and no of suicides cases")
#Firstly, we create a dataframe with the necessary features(columns) required for the operation.

data_year = df.groupby("year", as_index=False).suicides_no.sum()
data_year.head()# A total of 31 rows are there. we are showing just the first 5 rows to prevent complexcity.
# now we plot the above dataframe.

plt.figure(figsize=(10,5))
sns.barplot(x=data_year["year"], y=data_year["suicides_no"])
plt.title("showing the relation between year and no of suicides cases")
plt.xticks(rotation=45)
# Here we cross-checked the data of 2016.
a = df[df['year']==2016]
a.describe
data_gender = df.groupby("sex", as_index=False).suicides_no.sum()
data_gender
# To visualize the above data we use barplot.

plt.figure(figsize=(10,5))
sns.barplot(x=data_gender["sex"], y=data_gender["suicides_no"])
plt.title("Relation between gender and no of suicide cases")
# Creating a dataframe of generation and suicides_no.
data_gen = df.groupby("generation", as_index=False).suicides_no.sum()
data_gen
plt.figure(figsize=(10,5))
sns.barplot(x=data_gen["generation"], y=data_gen["suicides_no"])
plt.title("No of Suicide cases in each generation.")
# we create a dataframe with the necessary features.
data_country = df.groupby("country", as_index=False).suicides_no.sum()
data_country
# To make things easy, we plot the above values in a bargraph.
plt.figure(figsize=(100,100))
sns.barplot(x=data_country["suicides_no"], y=data_country["country"])
plt.title("Showing the no of suicides in every country")
plt.xticks(size = 50)
plt.yticks(size = 50)
#plt.xticks(rotation=80)
data_rate = df.groupby("country", as_index=False)["suicides/100k pop"].mean()
data_rate
# To make things easy, we plot the above values in a bargraph.
plt.figure(figsize=(100,100))
sns.barplot(x=data_rate["suicides/100k pop"], y=data_country["country"])
plt.title("Showing the suicides rates in every country")
plt.xticks(size = 50)
plt.yticks(size = 50)
# creating a dataframe.

data_gdp=df.groupby("country", as_index=False)["gdp_per_capita ($)"].mean()
data_gdp
data_s=df.groupby("country", as_index=False)["suicides_no"].sum()
data_s
#data_gdp.append(data_s, ignore_index=True)
concat_df = pd.concat([data_gdp, data_s], axis=1)
concat_df
plt.figure(figsize=(10,10))
sns.regplot(x=concat_df["gdp_per_capita ($)"], y=concat_df["suicides_no"]) 
plt.title("Relation between GDP per capita and no of suicide cases")