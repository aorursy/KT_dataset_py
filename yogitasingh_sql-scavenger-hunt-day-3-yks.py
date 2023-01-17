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
# Your code goes here :)
accidents.list_tables()

accidents.head("accident_2015")
#how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each hour
query = """SELECT COUNT(consecutive_number) as num_accidents, EXTRACT(DAYOFWEEK FROM timestamp_of_crash) as day_accident,
EXTRACT(HOUR FROM timestamp_of_crash) as hour_accident
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY day_accident, hour_accident
        ORDER BY  num_accidents DESC""" 

AccidentsPerHour = accidents.query_to_pandas_safe(query)
AccidentsPerHour.head()
accidents.head('vehicle_2015')
#Which state has the most hit and runs? Return a table with the number of vehicles registered in each 
#state that were involved in hit-and-run accidents, sorted by the number of hit and runs.
query ="""SELECT COUNT(consecutive_number) as num_hitAndruns, registration_state_name as state, hit_and_run
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        GROUP BY state, hit_and_run
        HAVING hit_and_run = "Yes"
        ORDER BY num_hitandruns DESC"""
HitsAndRuns = accidents.query_to_pandas_safe(query)
HitsAndRuns.head()
import matplotlib.pyplot as plt
plt.plot(AccidentsPerHour.num_accidents,AccidentsPerHour.hour_accident)
plt.title("Number of accidents each hour")
plt.plot( HitsAndRuns.state)
plt.title("Number of hit and runs for each state")