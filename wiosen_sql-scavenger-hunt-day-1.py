
import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
query1 = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'pmm' order by country
        """
countries = open_aq.query_to_pandas_safe(query1)
countries
query2 = """SELECT distinct pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00 order by pollutant
        """
pollutants = open_aq.query_to_pandas_safe(query2)
pollutants