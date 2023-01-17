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
accidents_by_day
# library for plotting
import matplotlib.pyplot as plt
#weekday_map= {2:'MON', 3:'TUE', 4:'WED', 5:'THU', 6:'FRI', 7:'SAT', 1:'SUN'}

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_,"ro")

plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
plt.ylabel("Number of accidents")
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
nhtsa_traffic_fatalities = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
nhtsa_traffic_fatalities.head("accident_2015")
# Your code goes here :)

query_hours = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hours = accidents.query_to_pandas_safe(query_hours)
accidents_by_hours.head()
accidents_by_hours.to_csv('accidents_by_hours.csv')
sns.set_style("whitegrid") 
ax = sns.barplot(x="f1_", y="f0_", data=accidents_by_hours,palette='coolwarm')
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
nhtsa_traffic_fatalities = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
nhtsa_traffic_fatalities.head("vehicle_2015")
# Your code goes here :)

query_hit_and_run = """SELECT registration_state_name, COUNT(hit_and_run) AS number_of_hit_and_run
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
states_hit_and_run = accidents.query_to_pandas_safe(query_hit_and_run)
states_hit_and_run.head()
# library for plotting
import matplotlib.pyplot as plt
import seaborn as sns

states_hit_and_run.to_csv("states_hit_and_run.csv")

f, ax = plt.subplots(figsize=(6, 15))
sns.set_style("whitegrid") 
ax = sns.barplot(x="number_of_hit_and_run", 
                 y="registration_state_name", 
                 data=states_hit_and_run,
                 palette='coolwarm',
                 dodge=False)
