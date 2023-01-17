# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# Which countries use a unit other than ppm to measure any type of pollution?
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            GROUP BY country
        """
q1 = open_aq.query_to_pandas_safe(query)

# print results for Q1
q1

# Which pollutants have a value of exactly 0?
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            GROUP BY pollutant
        """
q2 = open_aq.query_to_pandas_safe(query)

# print results for Q2
q2
