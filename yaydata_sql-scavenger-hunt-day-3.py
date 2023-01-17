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
# use a bar chart, not a line chart. 
plt.bar(accidents_by_day.index.values, accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# Examine the accidents dataset.
accidents.table_schema('accident_2015')
# Write and a run a query to answer question 1. 
query1 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash),
                COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
         """

# Estimate the query size in GB.
#accidents.estimate_query_size(query1)

# Run the query and output a pandas df.
accidents_by_hour = accidents.query_to_pandas_safe(query1)
# Graph accidents by hour, sorted by hour.
plt.figure(figsize=(12,8))
plt.scatter(accidents_by_hour.f0_, accidents_by_hour.f1_)
plt.title("Traffic Accidents by Hour")
plt.xlabel("Hour")
plt.ylabel("Count of Accidents")
print("The hour with the most accidents is hour",  accidents_by_hour.f0_[0]*100)
# Write and run a query to answer question 2.
query2 = """SELECT state_number, COUNT(DISTINCT(consecutive_number)) as hit_and_runs
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY state_number
            ORDER BY hit_and_runs DESC
         """

# Estimate the query size in GB.
#accidents.estimate_query_size(query2)

hit_and_run_by_state = accidents.query_to_pandas_safe(query2)
print(hit_and_run_by_state)
# Write and run a query to find the states with the most accidents.
query3 = """SELECT state_name, COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY state_name
            ORDER BY COUNT(consecutive_number) DESC
         """

# Estimate the query size in GB.
#accidents.estimate_query_size(query3)

accidents_by_state = accidents.query_to_pandas_safe(query3)
print(accidents_by_state)
# Graph accidents by state.
plt.figure(figsize=(8,20))
plt.xticks(rotation=90)
plt.barh(accidents_by_state.state_name, accidents_by_state.f0_)
plt.title("Total Accidents in 2015 by State")
