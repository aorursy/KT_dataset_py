# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Your code goes here :)
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
query1 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
ppm_countries = open_aq.query_to_pandas_safe(query1)
ppm_countries.country.value_counts().head()

query2= """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0"""

pollutantsZero = open_aq.query_to_pandas_safe(query2)
pollutantsZero.pollutant.value_counts().head()
import seaborn as sns
import matplotlib.pyplot as plt
# pollutant
plt.figure(figsize = (20,6))
sns.countplot(pollutantsZero['pollutant'])
# countries
query3 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
ppm_countries1 = open_aq.query_to_pandas_safe(query3)
ppm_countries1.country.value_counts().head()
plt.figure(figsize = (20,6))
sns.countplot(ppm_countries1['country'])
# saves ppm_countries1 and pollutantsZero
ppm_countries1.to_csv("ppm.csv")
pollutantsZero.to_csv("zero.csv")
# correlation
query4 = """SELECT * 
            FROM `bigquery-public-data.openaq.global_air_quality`"""
data_frame = open_aq.query_to_pandas_safe(query4)
data_frame.corr()
sns.heatmap(data_frame.corr(), yticklabels=False)