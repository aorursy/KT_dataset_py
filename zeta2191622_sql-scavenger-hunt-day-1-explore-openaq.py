# import package with helper functions 
import bq_helper
# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head("global_air_quality")
query_pllt = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit) != 'ppm'
            ORDER BY country
        """
countries = open_aq.query_to_pandas_safe(query_pllt)
print("There are {} countries not using PPM as unit".format(len(countries.country)))
print(list(countries.country))
query_zero_pll = """SELECT DISTINCT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant
        """
zero_pll = open_aq.query_to_pandas_safe(query_zero_pll)
print("There are {} pollutants that have value equal to zero".format(len(zero_pll)))
print(list(zero_pll.pollutant))
