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
query = """SELECT COUNT(consecutive_number) AS number, 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY hour
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
plt.plot(accidents_by_hour.hour, accidents_by_hour.number)
plt.title("Number of Accidents by hour of day \n")
plt.savefig("accidents_vs_hour.png")
query = """SELECT COUNTIF(hit_and_run = 'Yes') AS number, 
                  state_number
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            GROUP BY state_number
            ORDER BY number
        """
hitandrun_by_state = accidents.query_to_pandas_safe(query)
plt.bar(hitandrun_by_state.index, hitandrun_by_state.number, tick_label=hitandrun_by_state.state_number)
plt.title("Number of hit and runs by state \n")
plt.savefig("hitandrun_vs_state.png")