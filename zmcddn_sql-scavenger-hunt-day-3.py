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
# query to find out the number of accidents which 
# happen on each hour of the day for 2015
hour_query = """SELECT COUNT(consecutive_number), 
                EXTRACT(HOUR FROM timestamp_of_crash)
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(hour_query)
accidents_by_hour
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour each Day on 2015\n (Most to least dangerous)")
# query to find out the number of accidents which 
# happen on each hour of the day for 2016
hour_query2016 = """SELECT COUNT(consecutive_number), 
                EXTRACT(HOUR FROM timestamp_of_crash)
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour2016 = accidents.query_to_pandas_safe(hour_query2016)
plt.plot(accidents_by_hour2016.f0_)
plt.title("Number of Accidents by Rank of Hour each Day on 2016\n (Most to least dangerous)")
# plot the data and compare the number of crashes between year 2015 and 2016
l1 = plt.plot(accidents_by_hour.f0_, label='2015')
l2 = plt.plot(accidents_by_hour2016.f0_, label='2016')
plt.legend(loc='upper right')
plt.title("Number of Accidents by Rank of Hour each Day\n (Most to least dangerous)")
# Query for a table with the number of vehicles registered in each state that 
# were involved in hit-and-run accidents, sorted by the number of hit and runs. 
# Use either the vehicle_2015 or vehicle_2016 table for this, especially the 
# registration_state_name and hit_and_run columns.
hit_and_run_query2015 = """SELECT COUNT(hit_and_run), 
                registration_state_name
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                WHERE hit_and_run = 'Yes'
                GROUP BY registration_state_name
                ORDER BY COUNT(hit_and_run) DESC
        """
hit_and_runs2015 = accidents.query_to_pandas_safe(hit_and_run_query2015)
hit_and_runs2015
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 15))
# for plot details check https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
ax = sns.barplot(x="f0_", y="registration_state_name", data=hit_and_runs2015, palette='coolwarm', dodge=False)
hit_and_run_query2016 = """SELECT COUNT(hit_and_run), 
                registration_state_name
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                WHERE hit_and_run = 'Yes'
                GROUP BY registration_state_name
                ORDER BY COUNT(hit_and_run) DESC
        """
hit_and_runs2016 = accidents.query_to_pandas_safe(hit_and_run_query2016)
fig, ax = plt.subplots(figsize=(6, 15))
ax = sns.barplot(x="f0_", y="registration_state_name", data=hit_and_runs2016, palette='coolwarm', dodge=False)
# Compare the data for 2015 and 2016
import pandas as pd

df = pd.merge(hit_and_runs2015, hit_and_runs2016, on='registration_state_name', how='outer')
df