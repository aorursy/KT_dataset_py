# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.head("accident_2015")
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number) AS count, 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash) 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
accidents_by_day
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# Your code goes here :)

#QUestion1: Which hours of the day do the most accidents occur during?

MY_QUERY = """ SELECT EXTRACT(hour FROM timestamp_of_crash),
                COUNT(consecutive_number) AS Count_of_Accidents
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                ORDER BY Count_of_Accidents DESC
               
"""

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(MY_QUERY)
accidents_by_hour

#plotting the graph

import matplotlib.pyplot as plt

plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Hour of Day ")
#Question2: Which state has the most hit and runs?

MY_QUERY2 = """ SELECT COUNT(consecutive_number) AS No_of_hit_run, 
                    registration_state_name AS state
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                WHERE hit_and_run = "Yes"
                GROUP BY registration_state_name
                ORDER BY No_of_hit_run DESC
"""

#run the query
most_hit_run = accidents.query_to_pandas_safe(MY_QUERY2)
most_hit_run

#Answer- California tops the chart right now.


#plotting the chart

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))

plt.xticks(rotation='vertical')
plt.bar(most_hit_run.state, most_hit_run.No_of_hit_run, align='center', alpha=0.5)