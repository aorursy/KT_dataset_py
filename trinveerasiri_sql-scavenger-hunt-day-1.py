import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
# Your code goes here :)
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

unit_not_ppm = open_aq.query_to_pandas_safe(query)
unit_not_ppm.country.unique()
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

pollutant_zero = open_aq.query_to_pandas_safe(query)
pollutant_zero.pollutant.unique()