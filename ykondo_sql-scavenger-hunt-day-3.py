# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
query1 = """SELECT COUNT(consecutive_number) AS Accident_Count, 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS Hour
                  FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                  GROUP BY Hour
                  ORDER BY Accident_Count DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query1)
accidents_by_hour
list(accidents.head("vehicle_2015"))
query2 = """
        SELECT registration_state_name, COUNT(registration_state_name) as Count
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        WHERE hit_and_run = 'Yes'
        GROUP BY registration_state_name
        ORDER BY Count DESC
        """
hit_run = accidents.query_to_pandas_safe(query2)
hit_run