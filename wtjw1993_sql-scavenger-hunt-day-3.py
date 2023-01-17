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
query1 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as hour,
                   COUNT(consecutive_number) as number_of_accidents                   
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY number_of_accidents DESC
         """
accidents_by_hour = accidents.query_to_pandas_safe(query1)
accidents_by_hour
plt.plot(accidents_by_hour.sort_values('hour')['hour'], 
            accidents_by_hour.sort_values('hour')['number_of_accidents'])
plt.title('Number of Accidents by Hour of Day in 2015')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.show()
query2 = """SELECT registration_state_name,
                   COUNT(hit_and_run = 'Yes') as hit_and_runs
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
            ORDER BY hit_and_runs DESC
         """
hit_and_runs = accidents.query_to_pandas_safe(query2)
hit_and_runs.head()