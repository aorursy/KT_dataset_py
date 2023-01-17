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
# Preliminary exploration
accidents.list_tables()

# Get schema
accidents.table_schema("accident_2015")

query = """SELECT COUNT(EXTRACT(HOUR FROM timestamp_of_crash)), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(EXTRACT(HOUR FROM timestamp_of_crash)) DESC
        """
accidents_per_hour = accidents.query_to_pandas_safe(query)
accidents_per_hour     

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hit_and_run_by_state = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_per_hour.f1_)
plt.title("Accidents per Hour \n (Most to least)")
accidents.table_schema("accident_2016")
accidents.table_schema("vehicle_2016")

query = """SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            where hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
        
hit_and_run_by_state = accidents.query_to_pandas_safe(query)
hit_and_run_by_state


# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hit_and_run_by_state = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(hit_and_run_by_state.f0_)
plt.title("Hit and Run Accidents by State \n (Most to least)")