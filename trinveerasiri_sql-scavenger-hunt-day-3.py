# import package with helper functions 
import bq_helper
# library for plotting
import seaborn as sns
# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="nhtsa_traffic_fatalities")
accidents.head("accident_2015")
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
ax = sns.barplot(accidents_by_hour['f1_'], accidents_by_hour['f0_'], )
ax.set(xlabel='Hours', ylabel='Number of Accidents')
# Rename the column name
accidents_by_hour.columns = ['Number of Accidents', 'Hours']
accidents_by_hour
accidents.head("vehicle_2015")
query = """SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            where hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
hit_and_run = accidents.query_to_pandas_safe(query)
hit_and_run.head()