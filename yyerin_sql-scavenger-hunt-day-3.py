# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.table_schema('accident_2015')

# 1. Which hours of the day do the most accidents occur during?
query1 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS h, COUNT(consecutive_number) AS cnt
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY h 
            ORDER BY cnt DESC
            LIMIT 1
         """
accidents.query_to_pandas_safe(query1)

# 2. Which state has the most hit and runs?
accidents.table_schema('vehicle_2015')
query2 = """SELECT registration_state, COUNT(hit_and_run) AS cnt
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state
            ORDER BY cnt DESC
            LIMIT 1
         """
accidents.query_to_pandas_safe(query2)
