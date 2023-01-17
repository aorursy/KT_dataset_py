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
print(accidents_by_day)
fig, ax = plt.subplots()
#ax.yaxis.set_major_formatter(formatter)
import numpy as np
x = np.arange(7)

plt.bar(x, accidents_by_day.f0_)
plt.xticks(x, accidents_by_day.f1_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
plt.show()
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.list_tables()
# print the first couple rows of the "comments" table
accidents.head("accident_2016")

query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            
            --where consecutive_number in (190287,190247)
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            order by count(consecutive_number) desc
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
print(accidents_by_hour)
fig, ax = plt.subplots()
#ax.yaxis.set_major_formatter(formatter)
import numpy as np
x = np.arange(24)

plt.bar(x, accidents_by_hour.f0_)
plt.xticks(x, accidents_by_hour.f1_)
plt.title("Number of Accidents by Rank of Hour in 2016")
plt.show()
query = """SELECT COUNT(vehicle_number), 
                 registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            where hit_and_run = 'Yes' and registration_state_name != 'Unknown'
            group by registration_state_name
            ORDER BY COUNT(vehicle_number) DESC
        """
hit_and_run_by_state = accidents.query_to_pandas_safe(query)
print(hit_and_run_by_state)
fig, ax = plt.subplots(figsize=(15,15))
#ax.yaxis.set_major_formatter(formatter)
import numpy as np
x = np.arange(54)

plt.barh(x, hit_and_run_by_state.f0_)
plt.yticks(x, hit_and_run_by_state.registration_state_name)
plt.title("Number of Hit and Run Cars by State in 2016")
plt.show()