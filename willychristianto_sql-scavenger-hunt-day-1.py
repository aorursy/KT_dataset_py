






# import package with helper functions 
# same as example
import bq_helper

# create a helper object for this dataset
# same as example
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# 1st question is countries use unit other than PPM to measure any type of pollution
query = """SELECT distinct country FROM `bigquery-public-data.openaq.global_air_quality` WHERE unit != 'PPM' """
# the hint is using != to search data 'unit' other than ppm
# because we just want to get the countries so i change the sellect from City to country and with 
# using distinct i can make sure the same data will not appear because we just want to show the country

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
# Same as example
country_without_ppm = open_aq.query_to_pandas_safe(query)

# change this not same with the example because 
# the query have already in one field and we don't need to count it 
# the question is just about which country 
country_without_ppm.head()
# import package with helper functions 
# same as example
import bq_helper

# create a helper object for this dataset
# same as example
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# 2nd question is pollutant that have value exactly 0
query2 = """SELECT distinct pollutant, value FROM `bigquery-public-data.openaq.global_air_quality` WHERE value = 0 """
# the hint is 'value exactly 0' this is means we use 'value = 0'
# still using distinct because we don't need to show many data
# change the select to pollutant and value, to prove that this pollutant value is exactly 0
# change where condition to 'value = 0'

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
# Same as example
pollutant_exactly_0 = open_aq.query_to_pandas_safe(query2)

# change this not same with the example because 
# the question is just about which pollutant 
pollutant_exactly_0.head()