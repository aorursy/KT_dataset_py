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
query1="""SELECT country,unit
         FROM `bigquery-public-data.openaq.global_air_quality`
         where unit!='ppm'
"""
countries_munit = open_aq.query_to_pandas_safe(query1)

countries_munit.head()


#total number of observations staisfying the criteria
countries_munit.count()
# count of listed counties 
countries_munit.country.value_counts().count()

# count of units per country
countries_munit.groupby('country')['unit'].count()
# count of (unique) countries that have unit other than ppm
countries_munit.groupby('unit')['country'].nunique()
# country wise count of units other than ppm
countries_munit.country.value_counts().head()


query2="""SELECT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
where value =0"""
pollutant_value = open_aq.query_to_pandas_safe(query2)

pollutant_value.head()
pollutant_zero = pollutant_value.drop_duplicates(subset='pollutant').pollutant.tolist()
print('{} pollutants with value=0'.format(len(pollutant_zero) ))
print(pollutant_zero)
