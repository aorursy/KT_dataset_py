# import package with helper functions 

import bq_helper



# create a helper object for this dataset

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="nhtsa_traffic_fatalities")
# query to find out the number of accidents which 

# happen on each day of the week

query = """SELECT COUNT(consecutive_number) num_accidents, 

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

plt.plot(accidents_by_day.num_accidents)

plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")

plt.show()
print(accidents_by_day)