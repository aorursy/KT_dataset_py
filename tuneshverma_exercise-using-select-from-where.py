# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head("global_air_quality")

query="""SELECT country
         FROM `bigquery-public-data.openaq.global_air_quality`
         WHERE pollutant != 'pm25'"""
country_non_ppm=open_aq.query_to_pandas_safe(query)
#country_non_ppm
query_2=""" SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0"""
pollutant_value_zero=open_aq.query_to_pandas_safe(query_2)
pollutant_value_zero.pollutant.value_counts().head()
