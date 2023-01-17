
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head("global_air_quality")
queryCountry = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
country_units = open_aq.query_to_pandas_safe(queryCountry)
country_units.country.value_counts().head()
queryPollutants = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
Pollutant = open_aq.query_to_pandas_safe(queryPollutants)
Pollutant.pollutant.value_counts()