import bq_helper
import pandas as pd
import matplotlib.pylab as plt

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# accidents.table_schema('accident_2015')
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number) AS num_accidents, 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS weekday
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY weekday
            ORDER BY num_accidents DESC
        """
accidents.estimate_query_size(query)
accidents_by_day = accidents.query_to_pandas_safe(query)
accidents_by_day.head(10)
# Here I tried to create an ordered cathegorical index for weekday for nicer plotting
weekday_map={1:'Sunday',2:'Monday',3:'Tuesday',4:'Wednesday',5:'Thursday',6:'Friday',7:'Saturday'}
new_index=(pd.CategoricalIndex(accidents_by_day.weekday.map(weekday_map)).
           reorder_categories(new_categories=['Monday','Tuesday','Wednesday','Thursday',
                                              'Friday','Saturday','Sunday'],
                              ordered=True))
new_index
# I could do inplace but I don't want to loose the original data from the query
copy1=accidents_by_day.copy()
copy1.set_index(new_index,drop=True,inplace=True)
copy1
#  This should produce the same result as below
plt.plot(copy1.index.categories,copy1.num_accidents.values)

copy1.sort_index().num_accidents.plot(kind='bar')
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)");
# This gives me tuple index out of range which I don't underestand 
##plt.plot(copy1.num_accidents)
query = """SELECT COUNT(consecutive_number) AS num_accidents, 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY num_accidents DESC
        """
accidents.estimate_query_size(query)
accidents_by_hours=accidents.query_to_pandas_safe(query)
accidents_by_hours.head()
query = """SELECT COUNTIF(hit_and_run="Yes") AS num_hitrun, 
                  registration_state_name AS state
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY state
            ORDER BY num_hitrun DESC
        """
accidents.estimate_query_size(query)
vehicles_hAr_by_state=accidents.query_to_pandas_safe(query)
vehicles_hAr_by_state.head(10)