# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# to see the data description within the openaq dataser
open_aq.table_schema('global_air_quality')
#to check the first few rows of table
open_aq.head('global_air_quality')
#to view the first 10 rows of pollutant type
open_aq.head('global_air_quality', selected_columns= "pollutant", num_rows=10)
# Your Code Goes Here
# select columns- location,city,country (complete details), pollutant type and its unit where unit is not ppm
query1 = """ SELECT location, city, country, pollutant, unit
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE unit != 'ppm'
         """

#to estimate the size, to see if size is < 1GB
open_aq.estimate_query_size(query1)

#pandas dataframe
country_notppm = open_aq.query_to_pandas_safe(query1)
type(country_notppm)
#display first few rows of the dataframe
country_notppm.head(20)
# Your Code Goes Here

query2 = """ SELECT pollutant,value
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE value = 0
         """

#to estimate the size
open_aq.estimate_query_size(query2)
# return as dataframe
zerovalue_pollutant = open_aq.query_to_pandas_safe(query2)
#to check the number of unique pollutant types 
unique_zero_pollutant = zerovalue_pollutant.nunique()
unique_zero_pollutant
zerovalue_pollutant.head()