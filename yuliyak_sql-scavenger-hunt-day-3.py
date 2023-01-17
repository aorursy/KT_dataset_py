# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.head('accident_2015')
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
# Query to pass
query = """SELECT COUNT(consecutive_number),
                    EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# Run the query safely
hour_accidents = accidents.query_to_pandas_safe(query)
# Plot the data
plt.plot(hour_accidents.f0_)
plt.title("Number of Accidents by Rank of Day")
# Show the most dangerous hours
hour_accidents.head()
# Show first rows of a table
accidents.head("vehicle_2015")
accidents.table_schema("vehicle_2015")
# Query to pass
query = """SELECT COUNT(consecutive_number), registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
        """
# Run query safely
accidents_by_state = accidents.query_to_pandas_safe(query)
# Plot the results
plt.plot(accidents_by_state.f0_)
plt.title("Number of Hit-and-Runs Accidents by Rank of Registered State")
# Show first rows
accidents_by_state.head()